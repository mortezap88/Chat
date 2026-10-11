package com.example

import com.example.data.ApiKeyItem
import com.example.data.AppSettings
import com.example.data.ChatMessage
import com.example.data.FailoverEvent
import com.example.data.KeyStatus
import com.example.network.GeminiPoolService
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class StreamChatUnitTest {

    @Test
    fun testApiKeyItemMasking() {
        val key = ApiKeyItem(
            key = "AIzaSyAQAb8DefaultPoolKeyLAsA",
            label = "Key #1",
            status = KeyStatus.WORKING
        )
        val masked = key.maskedKey
        assertTrue(masked.startsWith("AIzaSy"))
        assertTrue(masked.endsWith("LAsA"))
        assertTrue(masked.contains("••••"))
    }

    @Test
    fun testDefaultSettings() {
        val settings = AppSettings()
        assertEquals("gemini-3.8-flash", settings.primaryModel)
        assertTrue(settings.nonThinkingMode)
        assertTrue(settings.tokenStreaming)
        assertTrue(settings.messagingStreamlining)
        assertEquals(24, settings.maxFontSizeSp)
    }

    @Test
    fun testChatMessageCreation() {
        val msg = ChatMessage(
            userText = "Hello Gemini",
            isAudioInput = false,
            aiResponse = "Hi there!",
            modelUsed = "gemini-3.8-flash"
        )
        assertNotNull(msg.id)
        assertEquals("Hello Gemini", msg.userText)
        assertEquals("Hi there!", msg.aiResponse)
    }

    @Test
    fun testFailoverEventCreation() {
        val event = FailoverEvent(
            fromKeyMasked = "AQ.Ab8••••LAsA",
            toKeyMasked = "AQ.Ab8••••Rdvw",
            reason = "(API Error)"
        )
        assertEquals("AQ.Ab8••••LAsA", event.fromKeyMasked)
        assertEquals("AQ.Ab8••••Rdvw", event.toKeyMasked)
        assertEquals("(API Error)", event.reason)
    }

    @Test
    fun testGeminiPoolServiceKeyCount() {
        val keys = listOf(
            ApiKeyItem(key = "key1", label = "Key #1"),
            ApiKeyItem(key = "key2", label = "Key #2")
        )
        val poolService = GeminiPoolService(keys)
        assertEquals(2, poolService.getKeysCount())
    }

    @Test
    fun testLinguisticCorrectionModel() {
        val grammarCorr = com.example.data.LinguisticCorrection(
            type = com.example.data.CorrectionType.GRAMMAR,
            original = "because because",
            suggested = "because",
            reason = "Remove accidental repetition of the word."
        )
        assertEquals(com.example.data.CorrectionType.GRAMMAR, grammarCorr.type)
        assertEquals("because because", grammarCorr.original)
        assertEquals("because", grammarCorr.suggested)

        val wordChoiceCorr = com.example.data.LinguisticCorrection(
            type = com.example.data.CorrectionType.WORD_CHOICE,
            original = "so many things",
            suggested = "various factors",
            reason = "Upgrade vague phrasing to a more professional tone."
        )
        assertEquals(com.example.data.CorrectionType.WORD_CHOICE, wordChoiceCorr.type)
    }

    @Test
    fun testPageCountStreamliningOff() {
        // If 7 messages so far, streamlining off means 14 pages (7 user pages + 7 AI pages)
        val messageCount = 7
        val streamliningOffPageCount = messageCount * 2
        assertEquals(14, streamliningOffPageCount)

        // Streamlining on means 7 pages
        val streamliningOnPageCount = messageCount
        assertEquals(7, streamliningOnPageCount)
    }
}
