# Prototype PII Filtering 
**CS 690F – Assignment 4**  
*by Kiet Chu*

---

## Dataset

**File:** `synthetic_student_behavior_100.csv`

Each row describes a student activity log that can contain personally identifiable information (PII).

**Main fields:**
- `subject_id` – identifier  
- `text` – texts containing PII  
- `pii` / `pii_parsed` – ground truth PII annotations  

**PII types found:**

| Type | Example |
|------|----------|
| NAME | Nicole Munoz |
| EMAIL | kimberly35@example.org |
| PHONE | 983-553-2053 |
| DOB | 2002-08-07 |
| CREDIT_CARD | 379526681030307 |
| IP_ADDRESS | 95.120.2.48 |
| SSN | 123-45-6789 |

---

## Design & Implementation

### Detector

A simple `PIIDetector` uses regex patterns and validation checks.

**Patterns:**
- Email: `[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}`
- Phone: `\d{3}-\d{3}-\d{4}`
- Credit card: 13–19 digits, validated with Luhn checksum
- Date of birth: `YYYY-MM-DD`
- SSN: `XXX-XX-XXXX`
- IP address: `123.123.123.123`
- Names: title-case pairs like “John Smith”

---

## Redaction Modes

| Mode | Description | Example |
|------|--------------|----------|
| StrictMask | Replace detected PII with `[TYPE]` tokens | `Nicole Munoz (kimberly35@example.org)` → `[NAME] ([EMAIL])` |
| PartialMask | Partially hide PII while preserving structure | `kimberly35@example.org` → `k***@example.org` |
| LLMMask (Qwen2.5-3B-Instruct) | Local small model that semantically redacts text | → `[EMAIL]`, `[NAME]`, etc. |

---

## Results

### Redaction Performance

| Redactor | Precision | Recall | F1 | Notes |
|-----------|------------|--------|----|-------|
| Strict | 1.00 | 1.00 | **1.00** | Fully replaces all PII |
| Partial | 1.00 | 1.00 | **1.00** | Equal protection, more readable |
| LLM (Qwen2.5-3B) | 1.00 | 0.81 | **0.88** | Caught most PII |

**Residual leakage:**
- None for strict or partial: This is because the pii in the synthetic data are easy to find through regex.
- Minor for LLM

---

## Adversarial Tests

10 adversarial examples with spacing, leetspeak, and obfuscation.

| Case | Example | Strict/Partial | LLM |
|------|----------|----------------|-----|
| Spaced phone | `5 5 5 - 1 2 3 - 4 5 6 7` | Missed | Caught |
| Leetspeak email | `j0hn.d03@ex4mpl3.com` | Caught | Caught |
| Unicode SSN | `123-45-6789` | Caught | Caught |
| Dotted credit card | `4.5.3.2.1.2.3.4...` | Caught | Caught |
| Mixed-case name | `JoHn SmItH` | Missed | Caught |
| Spaced IP | `1 9 2 . 1 6 8 . 1 . 1 0 0` | Missed | Caught |
| Parentheses phone | `(555) 123-4567` | Missed | Caught |
| Zero-padded date | `01/01/1990` | Missed | Caught |
| Hyphenated email | `john-doe@company-name.com` | Caught | Caught |
| Formatted card | `4532 1234 5678 9012` | Missed | Caught |

**Catch Rate Summary**

| Mode | Caught | Missed | Catch % |
|------|---------|--------|----------|
| Strict | 4 | 8 | 33.3% |
| Partial | 4 | 8 | 33.3% |
| LLM | 12 | 0 | **100%** |

---

## Takeaways

- Regex-based detection is fast and accurate for clean formats but fails under noise or irregular patterns. 
- LLM redaction doesn't perform as good as regex in conventional pii data, however, it outperforms in adversarial cases. This is due to the fact that LLMs can recognize more cases of pii without a predefined pattern.
- Stronger LLM can perform a lot better. While not shown in the result here, Qwen2.5-0.5B performed significantly worse and Qwen2.5-7B performed significantly better than Qwen2.5-3B did.

**Implication**
- Regex can be sufficient for the project since most students' PII are not adversarial.
- We can combine regex + LLM to improve the redaction performance.

**AI Usage**

I used ChatGPT to generate baseline codes for regex and conduct code review.
