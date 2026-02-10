# Essay Completion with `previous_parts` Parameter - Testing Guide

## Overview

This document describes how to test the new `previous_parts` parameter in the essay completion API that allows the MCP server to send previous essay sections directly instead of having ragrun load them from the filesystem.

## Changes Made

1. **ragrun API** (`app/retrieval/api/essay_completion.py`):
   - Added optional `previous_parts: Optional[str] = None` to `EssayCompletionRequest`
   - Parameter is passed through service → chain → graph layers

2. **ragrun Graph** (`app/retrieval/graphs/essay_completion.py`):
   - Modified `_load_previous_parts()` to accept `provided_parts` parameter
   - If `provided_parts` is provided, returns it immediately (bypasses filesystem)
   - For `mood_index >= 2`, missing/empty `previous_parts` now returns HTTP 400 (no filesystem fallback)

3. **ragprep MCP** (`src/mcp/essayMcpServer.ts`):
   - Added `readPreviousParts()` function in `essayTemplateStore.ts`
   - Updated `essay_complete_part` tool to read previous parts locally and include in API payload

## Test Scenarios

### Test 1: Missing `previous_parts` (Should Fail)

**Objective**: Verify that the API rejects requests for parts 2-7 without `previous_parts`.

**Steps**:
1. Start ragrun Docker container
2. Call essay-completion API without `previous_parts` parameter:

```bash
curl -X POST http://localhost:8000/api/v1/agent/philo-von-freisinn/graphs/essay-completion \
  -H "Content-Type: application/json" \
  -d '{
    "assistant": "philo-von-freisinn",
    "essay_slug": "test-essay",
    "essay_title": "Test Essay",
    "mood_index": 2,
    "current_text": "Current part text",
    "k": 5,
    "verbose": false,
    "force": false
  }'
```

**Expected**: API returns **HTTP 400** with message indicating `previous_parts` is required for `mood_index >= 2`.

### Test 2: With `previous_parts` Parameter (New Functionality)

**Objective**: Verify that API accepts and uses provided `previous_parts`.

**Steps**:
1. Start ragrun Docker container
2. Call essay-completion API with `previous_parts`:

```bash
curl -X POST http://localhost:8000/api/v1/agent/philo-von-freisinn/graphs/essay-completion \
  -H "Content-Type: application/json" \
  -d '{
    "assistant": "philo-von-freisinn",
    "essay_slug": "test-essay",
    "essay_title": "Test Essay",
    "mood_index": 2,
    "current_text": "Current part text",
    "previous_parts": "## Part 1 Header\n\nThis is part 1 text.",
    "k": 5,
    "verbose": false,
    "force": false
  }'
```

**Expected**: API should use the provided `previous_parts` text instead of loading from filesystem.

**Verification**: 
- Check logs to confirm no filesystem read attempted
- Verify generated essay part includes context from provided previous parts

### Test 3: MCP Tool End-to-End Test

**Objective**: Verify the full flow from MCP tool to ragrun API.

**Prerequisites**:
- ragrun Docker container running
- ragprep MCP server configured and running
- Essay file exists: `ragkeep/assistants/philo-von-freisinn/essays/test-essay.essay`

**Steps**:
1. Use MCP tool to complete part 2 of an essay:

```typescript
// Via MCP client (e.g., Cursor or other MCP-enabled tool)
{
  "tool": "essay_complete_part",
  "arguments": {
    "assistant": "philo-von-freisinn",
    "essay_slug": "test-essay",
    "mood_index": 2,
    "k": 5,
    "verbose": true
  }
}
```

**Expected**:
- MCP tool reads part 1 from local filesystem (ragkeep)
- MCP tool sends previous_parts to ragrun API
- ragrun uses provided previous_parts (not filesystem)
- Essay file is updated with generated part 2

**Verification**:
- Check MCP tool logs for "Reading previous parts locally"
- Check ragrun logs for "Using provided previous_parts"
- Verify essay file contains new part 2 content

### Test 4: Empty Previous Parts (mood_index = 1)

**Objective**: Verify handling when mood_index = 1 (no previous parts).

**Steps**:
1. Call API with mood_index=1 and no previous_parts:

```bash
curl -X POST http://localhost:8000/api/v1/agent/philo-von-freisinn/graphs/essay-completion \
  -H "Content-Type: application/json" \
  -d '{
    "assistant": "philo-von-freisinn",
    "essay_slug": "test-essay",
    "essay_title": "Test Essay",
    "mood_index": 1,
    "current_text": "",
    "k": 5
  }'
```

**Expected**: Should work without errors (mood_index=1 doesn't need previous parts).

### Test 5: MCP Tool with Empty Previous Parts

**Objective**: Verify MCP tool handles mood_index=1 correctly.

**Steps**:
```typescript
{
  "tool": "essay_complete_part",
  "arguments": {
    "assistant": "philo-von-freisinn",
    "essay_slug": "new-essay",
    "mood_index": 1
  }
}
```

**Expected**: 
- `readPreviousParts()` returns empty string
- API receives empty string for previous_parts
- Essay generation proceeds normally

### Test 6: Error Handling - Missing Previous Part

**Objective**: Verify proper error handling when a required previous part is missing.

**Prerequisites**:
- Essay file with only part 1 completed

**Steps**:
```typescript
{
  "tool": "essay_complete_part",
  "arguments": {
    "assistant": "philo-von-freisinn",
    "essay_slug": "incomplete-essay",
    "mood_index": 3  // Trying to generate part 3 when part 2 is missing
  }
}
```

**Expected**: 
- MCP tool throws error: "Part 2 is empty. All previous parts must be completed."
- User receives clear error message
- No API call is made to ragrun

## Manual Testing Checklist

- [ ] Test 1: Backward compatibility without previous_parts
- [ ] Test 2: API accepts and uses previous_parts
- [ ] Test 3: MCP tool end-to-end flow
- [ ] Test 4: Empty previous parts (mood_index=1)
- [ ] Test 5: MCP tool with mood_index=1
- [ ] Test 6: Error handling for missing parts

## Unit Tests

Run the unit tests:

```bash
cd /Users/michael/Reniets/Ai/ragrun
python -m pytest tests/test_essay_completion_previous_parts.py -v
```

**Note**: Unit tests require full environment with dependencies installed.

## Logs to Check

### ragrun logs:
```bash
docker logs ragrun-container
```

Look for:
- "Using provided previous_parts" (when parameter is provided)
- "Loading from filesystem" (when parameter is None)
- No errors related to essay file reading

### MCP logs:
Check Cursor/IDE console for MCP server output:
- "Reading previous parts locally"
- "Sending previous_parts to ragrun"

## Success Criteria

1. ✅ API works without `previous_parts` (backward compatible)
2. ✅ API works with `previous_parts` provided
3. ✅ MCP tool successfully reads and sends previous parts
4. ✅ No filesystem access from ragrun when previous_parts provided
5. ✅ Proper error handling for missing/empty parts
6. ✅ Essay generation quality unchanged

## Rollback Plan

If issues occur:
1. Set `previous_parts: null` in MCP tool payload
2. System will fall back to filesystem loading
3. Original functionality preserved
