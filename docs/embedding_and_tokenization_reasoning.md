# My reasoning: embedding choice and tokenization

## Shared vs separate embedding

At first I thought: shared embedding table for English + Igbo, and accept the
tradeoffs. My worry when this got questioned was about collisions: what if
English and Igbo have the same string but it means something different in
each language? That felt risky.

Then we checked the actual data. English and Igbo do share some exact-string
tokens (about 5,440 out of ~24k English / ~22k Igbo), but almost all of them
are proper nouns, numbers, or things like that — "London", "Tottenham", "MTN".
That's not a real collision. A proper noun should mean the same entity in
both languages anyway, so sharing that token isn't a problem.

So the collision worry turned out to be small. But I also had to correct
myself on the other side: I assumed sharing an embedding table would somehow
make the model learn that English "dog" and Igbo "nkịta" are related. That's
not true. Nothing in a shared embedding table forces two different strings to
end up close in vector space. My translation loss only tells the model
"produce the correct Igbo output" — it does not tell the model to push
embedding("dog") near embedding("nkịta"). Any cross-language mapping the
model learns happens in the LSTM's hidden state, not in the embedding table.

Given that:
- the collision risk is low (mostly harmless proper nouns)
- the semantic-alignment benefit I hoped for was never real anyway

the only actual benefit left from sharing is reusing embedding rows for that
~5,440 overlap, which is small. And sharing has a real cost: if encoder and
decoder use the same table/id-space, the decoder's final output layer has to
cover the whole combined vocab (~40k) instead of just the Igbo vocab
(~21.8k), even though the decoder will never legitimately need to output an
English word. That's wasted capacity for a benefit that's now known to be
small.

**Decision: separate embedding tables, one for English, one for Igbo, both
trained from scratch (not pretrained).** Simpler, and the small parameter-
sharing benefit isn't worth the extra complexity now that I know it was never
about semantic alignment.

## Word-level tokenization

My original plan: use word-level tokenization since the dataset is naturally
space-separated, then just check how it performs.

When I measured the raw data (splitting on whitespace only, no cleanup),
the numbers looked bad: over half the vocabulary on both sides shows up only
once in the training set, and validation OOV rate was 14.4% (English) and
10.9% (Igbo). But that measurement wasn't fair yet, because punctuation was
still glued onto words (like "2014." or "Yobe."), which inflates the
vocabulary artificially.

So the real plan is: normalize first — lowercase, and split punctuation off
as its own token — then measure vocab size / singleton rate / OOV rate again
on the cleaned version. Only after that do I decide word-level vs subword,
based on the real numbers instead of assuming.
