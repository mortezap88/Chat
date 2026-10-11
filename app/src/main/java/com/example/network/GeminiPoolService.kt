package com.example.network

import android.util.Log
import com.example.data.ApiKeyItem
import com.example.data.AppSettings
import com.example.data.FailoverEvent
import com.example.data.KeyStatus
import com.example.data.UnifiedLinguisticResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.TimeUnit

class GeminiPoolService(
    private var apiKeys: List<ApiKeyItem>,
    private var onKeyStatusChanged: (List<ApiKeyItem>) -> Unit = {},
    private var onFailover: (FailoverEvent) -> Unit = {}
) {
    private val client = OkHttpClient.Builder()
        .connectTimeout(60, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .build()

    private var activeKeyIndex = 0

    init {
        val firstUsable = apiKeys.indexOfFirst { it.status != KeyStatus.FAILED }
        if (firstUsable >= 0) {
            activeKeyIndex = firstUsable
        }
    }

    fun updateKeys(newKeys: List<ApiKeyItem>) {
        apiKeys = newKeys
        if (activeKeyIndex >= apiKeys.size) {
            activeKeyIndex = 0
        }
    }

    fun getActiveKeyIndex(): Int = activeKeyIndex
    fun getKeysCount(): Int = apiKeys.size

    suspend fun executeChatRequest(
        userText: String,
        audioBase64: String? = null,
        audioMimeType: String = "audio/mp4",
        settings: AppSettings,
        onTokenReceived: (String) -> Unit
    ): Result<String> = withContext(Dispatchers.IO) {
        if (apiKeys.isEmpty()) {
            return@withContext Result.failure(IllegalStateException("No API keys in pool. Please add a key in Settings."))
        }

        val totalAttempts = apiKeys.size
        var attemptsMade = 0
        var lastError: Throwable? = null

        val targetModel = resolveModelName(settings.primaryModel.ifBlank { "gemini-3.8-flash" })

        while (attemptsMade < totalAttempts) {
            val currentItem = apiKeys[activeKeyIndex]
            val currentKey = currentItem.key

            try {
                val fullResponse = try {
                    if (settings.tokenStreaming) {
                        callStreamApi(
                            apiKey = currentKey,
                            model = targetModel,
                            prompt = userText,
                            audioBase64 = audioBase64,
                            audioMime = audioMimeType,
                            nonThinking = settings.nonThinkingMode,
                            onTokenReceived = onTokenReceived
                        )
                    } else {
                        val res = callNonStreamApi(
                            apiKey = currentKey,
                            model = targetModel,
                            prompt = userText,
                            audioBase64 = audioBase64,
                            audioMime = audioMimeType,
                            nonThinking = settings.nonThinkingMode
                        )
                        onTokenReceived(res)
                        res
                    }
                } catch (e: Exception) {
                    // If targetModel gave 404 (not found), try standard preview model fallback
                    if (e.message?.contains("404") == true && targetModel != "gemini-3.5-flash") {
                        Log.i("GeminiPool", "Model $targetModel returned 404, attempting gemini-3.5-flash fallback")
                        if (settings.tokenStreaming) {
                            callStreamApi(
                                apiKey = currentKey,
                                model = "gemini-3.5-flash",
                                prompt = userText,
                                audioBase64 = audioBase64,
                                audioMime = audioMimeType,
                                nonThinking = settings.nonThinkingMode,
                                onTokenReceived = onTokenReceived
                            )
                        } else {
                            val res = callNonStreamApi(
                                apiKey = currentKey,
                                model = "gemini-3.5-flash",
                                prompt = userText,
                                audioBase64 = audioBase64,
                                audioMime = audioMimeType,
                                nonThinking = settings.nonThinkingMode
                            )
                            onTokenReceived(res)
                            res
                        }
                    } else {
                        throw e
                    }
                }

                updateKeyStatus(activeKeyIndex, KeyStatus.WORKING, null)
                return@withContext Result.success(fullResponse)

            } catch (e: Exception) {
                lastError = e
                val errorMsg = e.message ?: "Unknown error"
                Log.w("GeminiPool", "Key #${activeKeyIndex + 1} (${currentItem.maskedKey}) failed: $errorMsg")

                updateKeyStatus(activeKeyIndex, KeyStatus.FAILED, errorMsg)

                val prevKeyMasked = currentItem.maskedKey
                activeKeyIndex = (activeKeyIndex + 1) % apiKeys.size
                val nextKeyMasked = apiKeys[activeKeyIndex].maskedKey
                attemptsMade++

                onFailover(
                    FailoverEvent(
                        fromKeyMasked = prevKeyMasked,
                        toKeyMasked = nextKeyMasked,
                        reason = formatReason(errorMsg)
                    )
                )
            }
        }

        Result.failure(lastError ?: IllegalStateException("All $totalAttempts API keys failed."))
    }

    /**
     * Transcribes an audio chunk into text using the configured transcription model.
     */
    suspend fun transcribeAudioChunk(
        audioBase64: String,
        audioMimeType: String = "audio/mp4",
        preferredModel: String = "gemini-3.5-flash"
    ): Result<String> = withContext(Dispatchers.IO) {
        if (apiKeys.isEmpty()) {
            return@withContext Result.failure(IllegalStateException("No API keys available for transcription"))
        }

        val key = apiKeys[activeKeyIndex].key
        val model = resolveModelName(preferredModel)

        try {
            val res = callNonStreamApi(
                apiKey = key,
                model = model,
                prompt = "Transcribe this audio chunk verbatim into plain text. Output only the transcribed speech with accurate punctuation, without any extra commentary or quotation marks.",
                audioBase64 = audioBase64,
                audioMime = audioMimeType,
                nonThinking = true
            )
            Result.success(res.trim())
        } catch (e: Exception) {
            Log.w("GeminiPool", "Transcription chunk failed: ${e.message}")
            Result.failure(e)
        }
    }

    /**
     * Contextual smoothing: Takes previous context words (up to 20 words) + new chunk
     * and smooths punctuation and sentence boundaries seamlessly.
     */
    suspend fun smoothTextContextually(
        contextWords: String,
        newChunkText: String,
        preferredModel: String = "gemini-3.5-flash-lite"
    ): String = withContext(Dispatchers.IO) {
        if (apiKeys.isEmpty() || newChunkText.isBlank()) return@withContext newChunkText

        val key = apiKeys[activeKeyIndex].key
        val model = resolveModelName(preferredModel)

        try {
            val prompt = """
                You are a real-time speech boundary smoother.
                Previous text context: "$contextWords"
                New speech chunk to integrate: "$newChunkText"
                Task: Output ONLY the new speech chunk with appropriate capitalization, sentence boundary punctuation, and seamless flow following the previous context. Do not repeat the previous context words. Do not change words or add commentary.
            """.trimIndent()

            val res = callNonStreamApi(
                apiKey = key,
                model = model,
                prompt = prompt,
                audioBase64 = null,
                audioMime = "audio/wav",
                nonThinking = true
            )
            val cleaned = res.trim().removeSurrounding("\"")
            if (cleaned.isNotBlank()) cleaned else newChunkText
        } catch (e: Exception) {
            Log.w("GeminiPool", "Smoothing failed, using raw chunk: ${e.message}")
            newChunkText
        }
    }

    /**
     * Unified single-call smoothing + linguistic inspection:
     * Cleans up sentence boundaries and punctuation from multi-chunk speech recording,
     * while preserving spoken mistakes so they can be identified and highlighted with
     * Grammar (Red) and Word Choice (Yellow) correction elements in ONE single API call.
     */
    suspend fun inspectAndSmoothInSingleCall(
        rawUserText: String,
        preferredModel: String = "gemini-3.5-flash-lite"
    ): UnifiedLinguisticResult = withContext(Dispatchers.IO) {
        if (apiKeys.isEmpty() || rawUserText.isBlank()) {
            return@withContext UnifiedLinguisticResult(rawUserText, emptyList())
        }

        val key = apiKeys[activeKeyIndex].key
        val model = resolveModelName(preferredModel)

        try {
            val prompt = """
                You are a linguistic analyzer and transcription post-processor.
                Input user text: "$rawUserText"

                Perform both tasks in a single JSON response:
                1. "smoothedText": Clean up boundary punctuation, capitalization, and seams from speech chunk assembly. DO NOT fix grammatical mistakes, accidental repetitions (e.g. "because because"), or word choices in smoothedText—preserve the user's actual words so their mistakes remain visible for review.
                2. "corrections": Identify linguistic mistakes and improvements in the smoothedText:
                   - "type": "GRAMMAR" (for grammatical errors, misspellings, stutter repetitions) or "WORD_CHOICE" (for awkward, repetitive, or informal phrasing)
                   - "original": the exact substring from smoothedText
                   - "suggested": the improved replacement word or phrase
                   - "reason": a concise 1-sentence explanation of why

                Return ONLY a JSON object:
                {
                  "smoothedText": "...",
                  "corrections": [
                    {
                      "type": "GRAMMAR",
                      "original": "...",
                      "suggested": "...",
                      "reason": "..."
                    }
                  ]
                }
            """.trimIndent()

            val res = callNonStreamApi(
                apiKey = key,
                model = model,
                prompt = prompt,
                audioBase64 = null,
                audioMime = "audio/wav",
                nonThinking = true
            )

            val jsonStart = res.indexOf('{')
            val jsonEnd = res.lastIndexOf('}')
            if (jsonStart >= 0 && jsonEnd > jsonStart) {
                val jsonStr = res.substring(jsonStart, jsonEnd + 1)
                val rootObj = JSONObject(jsonStr)
                val smoothed = rootObj.optString("smoothedText", rawUserText).ifBlank { rawUserText }
                val corrArray = rootObj.optJSONArray("corrections")
                val list = mutableListOf<com.example.data.LinguisticCorrection>()

                if (corrArray != null) {
                    for (i in 0 until corrArray.length()) {
                        val obj = corrArray.getJSONObject(i)
                        val typeStr = obj.optString("type", "GRAMMAR")
                        val type = if (typeStr.equals("WORD_CHOICE", ignoreCase = true)) {
                            com.example.data.CorrectionType.WORD_CHOICE
                        } else {
                            com.example.data.CorrectionType.GRAMMAR
                        }
                        val original = obj.optString("original", "").trim()
                        val suggested = obj.optString("suggested", "").trim()
                        val reason = obj.optString("reason", "").trim()

                        if (original.isNotEmpty() && suggested.isNotEmpty()) {
                            list.add(
                                com.example.data.LinguisticCorrection(
                                    type = type,
                                    original = original,
                                    suggested = suggested,
                                    reason = reason
                                )
                            )
                        }
                    }
                }
                return@withContext UnifiedLinguisticResult(smoothed, list)
            }
            UnifiedLinguisticResult(rawUserText, emptyList())
        } catch (e: Exception) {
            Log.w("GeminiPool", "Unified inspection & smoothing failed: ${e.message}")
            UnifiedLinguisticResult(rawUserText, emptyList())
        }
    }

    private fun resolveModelName(model: String): String {
        return when (model.lowercase().trim()) {
            "gemini-3.8-flash" -> "gemini-3.8-flash"
            "gemini-3.5-flash" -> "gemini-3.5-flash"
            "gemini-3.5-flash-lite" -> "gemini-3.1-flash-lite-preview"
            "gemini-3.5-transcribe-live", "gemini-3.5-transcribe" -> "gemini-3.5-flash"
            else -> model
        }
    }

    private fun formatReason(errorMsg: String): String {
        return when {
            errorMsg.contains("400") -> "(API Error)"
            errorMsg.contains("404") -> "(Model Not Found)"
            errorMsg.contains("429") || errorMsg.contains("quota", ignoreCase = true) -> "(Quota Exceeded)"
            errorMsg.contains("403") || errorMsg.contains("permission", ignoreCase = true) -> "(Forbidden / Invalid Key)"
            errorMsg.contains("timeout", ignoreCase = true) -> "(Connection Timeout)"
            else -> "(API Error)"
        }
    }

    private fun updateKeyStatus(index: Int, status: KeyStatus, errorMsg: String?) {
        val updated = apiKeys.toMutableList()
        val item = updated[index]
        updated[index] = item.copy(
            status = status,
            errorMessage = errorMsg,
            failCount = if (status == KeyStatus.FAILED) item.failCount + 1 else item.failCount
        )
        apiKeys = updated
        onKeyStatusChanged(updated)
    }

    private fun buildRequestBody(
        prompt: String,
        audioBase64: String?,
        audioMime: String,
        nonThinking: Boolean
    ): String {
        val root = JSONObject()
        val contentsArray = JSONArray()
        val contentObj = JSONObject()
        val partsArray = JSONArray()

        if (!audioBase64.isNullOrEmpty()) {
            val audioPart = JSONObject()
            val inlineData = JSONObject()
            inlineData.put("mimeType", audioMime)
            inlineData.put("data", audioBase64)
            audioPart.put("inlineData", inlineData)
            partsArray.put(audioPart)

            val textInstruction = JSONObject()
            textInstruction.put("text", if (prompt.isNotBlank()) prompt else "Listen to this audio and respond conversationally and concisely.")
            partsArray.put(textInstruction)
        } else {
            val textPart = JSONObject()
            textPart.put("text", prompt)
            partsArray.put(textPart)
        }

        contentObj.put("parts", partsArray)
        contentsArray.put(contentObj)
        root.put("contents", contentsArray)

        val generationConfig = JSONObject()
        // For non-thinking mode: omit thinkingConfig to eliminate thinking delays completely
        if (!nonThinking) {
            val thinkingConfig = JSONObject()
            thinkingConfig.put("thinkingLevel", "low")
            generationConfig.put("thinkingConfig", thinkingConfig)
        }
        generationConfig.put("temperature", 0.7)
        root.put("generationConfig", generationConfig)

        return root.toString()
    }

    private fun callStreamApi(
        apiKey: String,
        model: String,
        prompt: String,
        audioBase64: String?,
        audioMime: String,
        nonThinking: Boolean,
        onTokenReceived: (String) -> Unit
    ): String {
        val url = "https://generativelanguage.googleapis.com/v1beta/models/$model:streamGenerateContent?alt=sse&key=$apiKey"
        val jsonPayload = buildRequestBody(prompt, audioBase64, audioMime, nonThinking)
        val body = jsonPayload.toRequestBody("application/json".toMediaType())

        val request = Request.Builder()
            .url(url)
            .post(body)
            .build()

        val response = client.newCall(request).execute()
        if (!response.isSuccessful) {
            val errorBody = response.body?.string() ?: ""
            val parsedError = parseApiErrorMessage(response.code, errorBody)
            throw IllegalStateException(parsedError)
        }

        val responseBody = response.body ?: throw IllegalStateException("Empty response body from Gemini API")
        val source = responseBody.source()
        val accumulatedText = StringBuilder()

        try {
            while (!source.exhausted()) {
                val current = source.readUtf8Line() ?: break
                val trimmed = current.trim()
                if (!trimmed.startsWith("data:")) continue
                val payload = trimmed.removePrefix("data:").trim()
                if (payload.isEmpty() || payload == "[DONE]") continue

                try {
                    val jsonObj = JSONObject(payload)
                    val candidates = jsonObj.optJSONArray("candidates")
                    if (candidates != null && candidates.length() > 0) {
                        val firstCandidate = candidates.getJSONObject(0)
                        val content = firstCandidate.optJSONObject("content")
                        val parts = content?.optJSONArray("parts")
                        if (parts != null && parts.length() > 0) {
                            val part = parts.getJSONObject(0)
                            val token = part.optString("text", "")
                            if (token.isNotEmpty()) {
                                accumulatedText.append(token)
                                onTokenReceived(accumulatedText.toString())
                            }
                        }
                    }
                } catch (e: Exception) {
                    Log.v("GeminiPool", "Chunk parse skip: ${e.message}")
                }
            }
        } finally {
            source.close()
        }

        val result = accumulatedText.toString()
        if (result.isEmpty()) {
            throw IllegalStateException("Received empty stream response from $model")
        }
        return result
    }

    private fun callNonStreamApi(
        apiKey: String,
        model: String,
        prompt: String,
        audioBase64: String?,
        audioMime: String,
        nonThinking: Boolean
    ): String {
        val url = "https://generativelanguage.googleapis.com/v1beta/models/$model:generateContent?key=$apiKey"
        val jsonPayload = buildRequestBody(prompt, audioBase64, audioMime, nonThinking)
        val body = jsonPayload.toRequestBody("application/json".toMediaType())

        val request = Request.Builder()
            .url(url)
            .post(body)
            .build()

        val response = client.newCall(request).execute()
        if (!response.isSuccessful) {
            val errorBody = response.body?.string() ?: ""
            val parsedError = parseApiErrorMessage(response.code, errorBody)
            throw IllegalStateException(parsedError)
        }

        val responseBody = response.body?.string() ?: throw IllegalStateException("Empty response body")
        val jsonObj = JSONObject(responseBody)
        val candidates = jsonObj.optJSONArray("candidates")
        if (candidates != null && candidates.length() > 0) {
            val candidate = candidates.getJSONObject(0)
            val content = candidate.optJSONObject("content")
            val parts = content?.optJSONArray("parts")
            if (parts != null && parts.length() > 0) {
                return parts.getJSONObject(0).optString("text", "")
            }
        }
        return "No text response generated."
    }

    private fun parseApiErrorMessage(code: Int, body: String): String {
        try {
            val json = JSONObject(body)
            val errObj = json.optJSONObject("error")
            if (errObj != null) {
                val message = errObj.optString("message", "")
                if (message.isNotEmpty()) {
                    return "Gemini API error ($code): $message"
                }
            }
        } catch (_: Exception) {}
        return "Gemini API error ($code)"
    }
}
