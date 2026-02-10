# Contextual Integrity Evaluation
**CS 690F – Assignment 5**
*Team 3: Intelligent Tutoring Systems R Us (Chase Randall, lead)*

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


### Contextual Integrity Evaluation

<img width="692" height="256" alt="image" src="https://github.com/user-attachments/assets/6f7c1141-fc94-490c-b016-92d76a6700ca" />

**Interpretation**
- Lower leak rate = better protectoin.
- Both `airgap_partial` and `airgap_strict` prevented all PII leaks ( 0.0 ).  
- The naive baseline had full exposure ( 1.0 ).  
- Strict and Partial masking both achieved perfect contextual integrity scores.

**Saved output:** `a5_contextual_integrity_results.csv`

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

### Contextual Integrity Framework

This assignment extends A4 by measuring **privacy leakage** under simulated policy contexts.

- **Direct leak:** When PII is disclosed without justification.  
- **Good-cause:** When limited PII is shared within contextually appropriate bounds (e.g., for grading or support).  
- **Leak rate metric:** Fraction of samples where PII is improperly revealed.  

---
## Takeaways

- Regex-based and partial mask approaches maintain privacy perfectly on synthetic data.  
- Both strict and partial redactors preserved contextual integrity (F1 = 1.0, leak rate = 0).  
- The framework can be extended to evaluate real chatbot defenses against policy violations.  

**Next Steps**
- Integrate context policies more explicitly (e.g., student–teacher vs admin contexts).  
- Compare LLM-based redactors once model loading issues are resolved.

- Regex-based detection is fast and accurate for clean formats but fails under noise or irregular patterns. 
- LLM redaction doesn't perform as good as regex in conventional pii data, however, it outperforms in adversarial cases. This is due to the fact that LLMs can recognize more cases of pii without a predefined pattern.
- Stronger LLM can perform a lot better. While not shown in the result here, Qwen2.5-0.5B performed significantly worse and Qwen2.5-7B performed significantly better than Qwen2.5-3B did.

**Implication**
- Perfect contextual integrity (0 leak rate) means the redactors not only detect PII but also respect when and how information should be shared.  
- Strict and partial masking can serve as reliable defenses in educational AI systems that must enforce data minimization principles.  
- While regex-based methods are enough for synthetic text, real educational data may include contextual cues (e.g., role, purpose, consent) that require reasoning beyond pattern matching.  
- Future iterations could integrate policy-aware large language models to dynamically decide when disclosure is permissible under contextual integrity norms.


**AI Usage**

I borrowed code from teammate (assignment #4) who used AI for completion.  In this assignmnet, I used AI as sanity check for evaluation code for contextual integrity leak rate analysis and otherwise as a means to trouble shoot errors.
