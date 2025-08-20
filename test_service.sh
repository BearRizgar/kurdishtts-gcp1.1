#!/bin/bash

# Test script for Kurdish TTS Cloud Run Service
SERVICE_URL="https://kurdishtts-gcp-137160846844.europe-north2.run.app"

echo "🧪 Testing Kurdish TTS Cloud Run Service"
echo "Service URL: $SERVICE_URL"
echo ""

# Test 1: Check Sorani speakers
echo "1️⃣ Testing Sorani speakers endpoint..."
SORANI_RESPONSE=$(curl -s "$SERVICE_URL/speakers?dialect=sorani")
SORANI_COUNT=$(echo "$SORANI_RESPONSE" | jq -r '.available_speakers | length' 2>/dev/null || echo "0")
echo "   ✅ Sorani speakers available: $SORANI_COUNT"
echo ""

# Test 2: Check Kurmanji speakers
echo "2️⃣ Testing Kurmanji speakers endpoint..."
KURMANJI_RESPONSE=$(curl -s "$SERVICE_URL/speakers?dialect=kurmanji")
KURMANJI_COUNT=$(echo "$KURMANJI_RESPONSE" | jq -r '.available_speakers | length' 2>/dev/null || echo "0")
echo "   ✅ Kurmanji speakers available: $KURMANJI_COUNT"
echo ""

# Test 3: Generate Sorani audio
echo "3️⃣ Testing Sorani TTS generation..."
curl -s -X POST "$SERVICE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{"text": "سڵاو، چۆنی؟", "dialect": "sorani", "speaker": "speaker_0000"}' \
  --output test_sorani.wav
SORANI_SIZE=$(ls -la test_sorani.wav 2>/dev/null | awk '{print $5}' || echo "0")
echo "   ✅ Sorani audio generated: ${SORANI_SIZE} bytes"
echo ""

# Test 4: Generate Kurmanji audio
echo "4️⃣ Testing Kurmanji TTS generation..."
curl -s -X POST "$SERVICE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{"text": "Silav, çawa yî?", "dialect": "kurmanji", "speaker": "speaker_0000"}' \
  --output test_kurmanji.wav
KURMANJI_SIZE=$(ls -la test_kurmanji.wav 2>/dev/null | awk '{print $5}' || echo "0")
echo "   ✅ Kurmanji audio generated: ${KURMANJI_SIZE} bytes"
echo ""

# Test 5: Verify audio files
echo "5️⃣ Verifying audio files..."
if [ -f "test_sorani.wav" ] && [ -f "test_kurmanji.wav" ]; then
    echo "   ✅ Both audio files created successfully"
    echo "   📁 Files: test_sorani.wav (${SORANI_SIZE} bytes), test_kurmanji.wav (${KURMANJI_SIZE} bytes)"
else
    echo "   ❌ Audio file creation failed"
fi
echo ""

# Summary
echo "📊 Test Summary:"
echo "   • Sorani speakers: $SORANI_COUNT"
echo "   • Kurmanji speakers: $KURMANJI_COUNT"
echo "   • Sorani audio: ${SORANI_SIZE} bytes"
echo "   • Kurmanji audio: ${KURMANJI_SIZE} bytes"
echo ""

if [ "$SORANI_COUNT" -gt 0 ] && [ "$KURMANJI_COUNT" -gt 0 ] && [ "$SORANI_SIZE" -gt 0 ] && [ "$KURMANJI_SIZE" -gt 0 ]; then
    echo "🎉 All tests passed! Your service is working correctly."
else
    echo "⚠️  Some tests failed. Check the service logs for issues."
fi

echo ""
echo "🔗 Service URL: $SERVICE_URL"
echo "📝 To play audio files: open test_sorani.wav and test_kurmanji.wav in your audio player" 