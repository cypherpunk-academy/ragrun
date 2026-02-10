# Essay Tune Part - Manual Testing Guide

## Prerequisites

1. Start the ragrun server:
```bash
cd /Users/michael/Reniets/Ai/ragrun
PYTHONPATH="." python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8010 --reload
```

2. Verify the ragprep MCP server is configured in Cursor settings.

## Test 1: Direct API Test

Test the API endpoint directly using curl:

```bash
curl -X POST "http://127.0.0.1:8010/api/v1/agent/philo-von-freisinn/graphs/essay-tune-part" \
  -H "Content-Type: application/json" \
  -d '{
    "assistant": "philo-von-freisinn",
    "essay_slug": "manchmal-hilft-es-zu-luegen",
    "essay_title": "manchmal-hilft-es-zu-luegen",
    "mood_index": 1,
    "mood_name": "okkult",
    "current_text": "Stell dir vor, du stehst auf einem festen Boden...",
    "current_header": "Wenn die Wahrheit schmerzt, die Lüge heilt",
    "previous_parts": "",
    "modifications": "Mache den Text persönlicher und füge ein konkretes Beispiel hinzu",
    "k": 5,
    "verbose": true,
    "retries": 3
  }'
```

Expected response: JSON with `revised_text`, `revised_header`, and reference lists.

## Test 2: MCP Command Test

Test via the MCP interface in Cursor:

```typescript
// Call via Cursor MCP
essay_tune_part({
  assistant: "philo-von-freisinn",
  essay_slug: "manchmal-hilft-es-zu-luegen",
  mood_index: 1,
  modifications: "Mache den Text persönlicher und füge ein Beispiel hinzu",
  k: 5,
  verbose: true
})
```

Expected behavior:
1. Reads current part 1 from the essay file
2. Reads previous parts (empty for part 1)
3. Sends request to ragrun API
4. Updates the essay file with revised text and header
5. Returns success status with updated response

## Test 3: Verify File Updates

After running the MCP command, check that the essay file was updated:

```bash
cat /Users/michael/Reniets/Ai/ragkeep/assistants/philo-von-freisinn/essays/manchmal-hilft-es-zu-luegen.essay
```

Verify that:
- Part 1 header is updated
- Part 1 text is updated with modifications applied
- The modifications reflect the instruction given

## Test 4: Test with Different Parts

Test with part 2 (which requires previous_parts):

```typescript
essay_tune_part({
  assistant: "philo-von-freisinn",
  essay_slug: "manchmal-hilft-es-zu-luegen",
  mood_index: 2,
  modifications: "Füge eine präzisere philosophische Argumentation hinzu",
  k: 5,
  verbose: false
})
```

## Test 5: Error Handling

Test error cases:

1. Missing modifications:
```typescript
essay_tune_part({
  assistant: "philo-von-freisinn",
  essay_slug: "manchmal-hilft-es-zu-luegen",
  mood_index: 1,
  // modifications missing - should fail
})
```

2. Invalid mood_index:
```typescript
essay_tune_part({
  assistant: "philo-von-freisinn",
  essay_slug: "manchmal-hilft-es-zu-luegen",
  mood_index: 9,  // Invalid - should fail
  modifications: "Test"
})
```

## Verification Checklist

- [ ] API endpoint responds at `/agent/philo-von-freisinn/graphs/essay-tune-part`
- [ ] MCP command `essay_tune_part` is available in Cursor
- [ ] Primary book retrieval works (k chunks from primary books)
- [ ] Secondary book retrieval works (k chunks from secondary books)
- [ ] Prompt template is correctly loaded and rendered
- [ ] LLM generates revised text based on modifications
- [ ] Header is generated for revised text
- [ ] Essay file is updated with new text and header
- [ ] Previous parts context is included for parts 2-7
- [ ] Error handling works for invalid inputs
- [ ] Verbose logging works when enabled

## Architecture Verification

Verify the data flow:

1. MCP Server (ragprep) → reads essay file, calls ragrun API
2. API Layer (ragrun) → validates request, calls service
3. Service → calls graph
4. Graph → retrieves from Qdrant (primary + secondary), builds prompt, calls LLM
5. Response → returns to MCP server
6. MCP Server → updates essay file

## Success Criteria

The test is successful if:
1. All API calls complete without errors
2. Essay file is updated with modified text
3. The modified text reflects the given instructions
4. Primary and secondary context is used in generation
5. References are returned in the response
