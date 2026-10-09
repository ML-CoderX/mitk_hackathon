# Context Surgeon dataset

40 AI-assisted synthetic examples with fictional entities: 10 conversations,
10 documents, 10 code examples and 10 structured-data examples.

Each record contains reference context, a question, expected answer, required
evidence substrings, challenge, and a soft estimated compression budget.
30 records are development cases and 10 are reserved. The reserved split is
a development convention, not an independently authored hidden test set.

Includes corrections, negation, missing information, paraphrases, units, arithmetic,
code dependencies, NULL, leading zeroes, table joins and one Hindi example.
Required facts are verbatim evidence, not computed answers. Every fact is checked
against its source context when generating the dataset.

Limitations: small, synthetic, mostly short, English-dominant and not representative
of production traffic. Fact-substring retention is a debugging metric, not answer
correctness. No independent annotation or external validation is claimed.
Manually score model answers before reporting answer quality. Keep the SHA256
file with evaluation reports to identify the dataset version.
