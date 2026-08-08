import json
import os
from collections import defaultdict

# PATHS
JSONL_DIR = r"D:\InternshipGU\Assamese_instrument_VLM\QWEN_2.5_VL3B\jsonl"
IMAGE_ROOT = r"D:\InternshipGU\Assamese_instrument_VLM\QWEN_2.5_VL3B"
SYSTEM_TEXT = "You are an expert on Assamese musical instruments."

EXPECTED_COUNTS = {"train": 1386, "validation": 315, "test": 315}
EXPECTED_IMAGES = {"train": 154, "validation": 35, "test": 35}

passed = 0
failed = 0

def ok(msg):
    global passed
    passed += 1
    print(f"  ✅ {msg}")

def fail(msg):
    global failed
    failed += 1
    print(f"  ❌ {msg}")

# LOAD ALL SPLITS
all_data = {}
all_images_per_split = {}

for split in ["train", "validation", "test"]:
    path = os.path.join(JSONL_DIR, f"{split}.jsonl")

    # Check 1: File exists
    if not os.path.exists(path):
        fail(f"{split}.jsonl does not exist")
        all_data[split] = []
        continue
    ok(f"{split}.jsonl exists")

    samples = []
    parse_errors = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                fail(f"{split}.jsonl line {i+1}: JSON parse error — {e}")
                parse_errors += 1

    if parse_errors == 0:
        ok(f"{split}.jsonl — all lines valid JSON")

    all_data[split] = samples

    # Check 2: Sample counts
    if len(samples) == EXPECTED_COUNTS[split]:
        ok(f"{split} sample count: {len(samples)}")
    else:
        fail(f"{split} sample count: expected {EXPECTED_COUNTS[split]}, got {len(samples)}")

# PER-SAMPLE CHECKS
all_keys = set()
image_to_splits = defaultdict(set)

for split, samples in all_data.items():
    image_qa_count = defaultdict(int)
    unique_images = set()
    duplicate_count = 0

    for i, sample in enumerate(samples):
        loc = f"{split} sample {i+1}"

        # Check 4: messages key exists
        if "messages" not in sample:
            fail(f"{loc}: missing 'messages' key")
            continue

        messages = sample["messages"]

        # Check 4: exactly 3 messages
        if len(messages) != 3:
            fail(f"{loc}: expected 3 messages, got {len(messages)}")
            continue

        roles = [m.get("role") for m in messages]
        if roles != ["system", "user", "assistant"]:
            fail(f"{loc}: wrong roles — {roles}")
            continue

        system_msg, user_msg, assistant_msg = messages

        # Check 5: system message
        sys_content = system_msg.get("content", [])
        if len(sys_content) != 1:
            fail(f"{loc}: system should contain exactly one content item")

        if sys_content[0].get("type") != "text":
            fail(f"{loc}: system content type should be 'text'")

        if sys_content and sys_content[0].get("text") == SYSTEM_TEXT:
            pass  # ok, checked in aggregate
        else:
            fail(f"{loc}: wrong system message — '{sys_content}'")

        # Check 6: user has exactly 2 contents
        user_content = user_msg.get("content", [])
        if len(user_content) != 2:
            fail(f"{loc}: user content should have 2 items, got {len(user_content)}")
            continue

        user_types = [c.get("type") for c in user_content]
        if user_types != ["image", "text"]:
            fail(f"{loc}: user content types wrong — {user_types}")

        image_block = user_content[0]
        text_block  = user_content[1]
        if image_block.get("type") != "image":
            fail(f"{loc}: first user content should be image")

        if text_block.get("type") != "text":
            fail(f"{loc}: second user content should be text")

        # Check 7: image
        image_path = image_block.get("image", "")
        if not image_path:
            fail(f"{loc}: image field empty")
        else:
            # Check 14: relative path starts with dataset_32images/
            if not image_path.startswith("dataset_32images/"):
                fail(f"{loc}: image path doesn't start with 'dataset_32images/' — {image_path}")

            # Check 7: image exists on disk
            full_path = os.path.join(IMAGE_ROOT, image_path.replace("/", os.sep))
            if not os.path.exists(full_path):
                fail(f"{loc}: image not found on disk — {full_path}")

        # Check 8: question
        question = text_block.get("text", "")
        if not isinstance(question, str) or not question.strip():
            fail(f"{loc}: question empty or null")

        # Check 9: assistant answer
        assistant_content = assistant_msg.get("content", [])
        if len(assistant_content) != 1:
            fail(f"{loc}: assistant should contain exactly one content item")

        if assistant_content and assistant_content[0].get("type") != "text":
            fail(f"{loc}: assistant content type should be 'text'")

        if not assistant_content:
            fail(f"{loc}: assistant content empty")
        else:
            answer = assistant_content[0].get("text", "")
            if not isinstance(answer, str) or not answer.strip():
                fail(f"{loc}: answer empty or null")

        # Check 15: unicode preserved
        combined = question + answer + image_path
        try:
            combined.encode("utf-8").decode("utf-8")
        except Exception as e:
            fail(f"{loc}: unicode error — {e}")

        # Check 10: duplicates
        key = (image_path, question)
        if key in all_keys:
            duplicate_count += 1
        all_keys.add(key)

        # For check 11: split leakage
        image_to_splits[image_path].add(split)

        # For check 12 & 13
        unique_images.add(image_path)
        image_qa_count[image_path] += 1

    if duplicate_count == 0:
        ok(f"{split}: no duplicate (image+question) pairs")
    else:
        fail(f"{split}: {duplicate_count} duplicate (image+question) pairs found")

    # Check 12: unique image count
    if len(unique_images) == EXPECTED_IMAGES[split]:
        ok(f"{split}: unique image count = {len(unique_images)}")
    else:
        fail(f"{split}: unique image count expected {EXPECTED_IMAGES[split]}, got {len(unique_images)}")

    # Check 13: 9 QA pairs per image
    bad_images = [img for img, count in image_qa_count.items() if count != 9]
    if not bad_images:
        ok(f"{split}: every image has exactly 9 QA pairs")
    else:
        fail(f"{split}: {len(bad_images)} images don't have exactly 9 QA pairs — {bad_images[:3]}")

# Check 11: split leakage
leaking = [img for img, splits in image_to_splits.items() if len(splits) > 1]
if not leaking:
    ok("No split leakage — no image appears in multiple splits")
else:
    fail(f"Split leakage detected: {leaking[:3]}")

# FINAL SUMMARY
print("\n" + "="*50)
print(f"PASSED: {passed}  |  FAILED: {failed}")
if failed == 0:
    print("✅ JSONL dataset is production-ready for Qwen2.5-VL fine-tuning.")
else:
    print("❌ Fix the above errors before training.")
print("="*50)