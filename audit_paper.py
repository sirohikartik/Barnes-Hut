import re

with open("main.tex") as f:
    text = f.read()

lines = [l.split("%")[0] for l in text.splitlines()]
clean_text = "\n".join(lines)

# 1. Em-dashes
em_dashes = re.findall(r"---|—", clean_text)
print(f"Em-dashes found: {len(em_dashes)}")

# 2. (e.g., ...)
eg_matches = re.findall(r"\(e\.g\.", clean_text, re.IGNORECASE)
print(f"(e.g., ...) found: {len(eg_matches)}")

# 3. Banned words
banned = ["delve", "tapestry", "landscape", "pivotal", "crucial", "highlighting", "unravel", "testament", "beacon", "multifaceted", "plethora", "myriad"]
for b in banned:
    matches = re.findall(rf"\b{b}\w*\b", clean_text, re.IGNORECASE)
    if matches:
        print(f"Banned word '{b}' found: {matches}")

# 4. Banned openers
banned_openers = ["Moreover", "Furthermore", "Additionally", "Notably", "Importantly", "Intriguingly"]
for b in banned_openers:
    matches = re.findall(rf"(?:^|\.\s+){b}\b", clean_text)
    if matches:
        print(f"Banned opener '{b}' found: {matches}")

# 5. Sentences opening with prepositions
preps = ["In", "For", "With", "Under", "Across", "By", "Through", "During", "On", "At", "From", "Within", "Throughout"]
sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", clean_text)
prep_violations = []
long_sentences = []

for s in sentences:
    # ignore environments, tables, equations, itemize
    if s.strip().startswith("\\") or s.strip().startswith("&") or s.strip().startswith("}"):
        continue
    s_clean = re.sub(r"\\[a-zA-Z]+(\[[^\]]*\])?(\{[^\}]*\})*", "", s)
    s_clean = re.sub(r"\$[^$]*\$", "X", s_clean)
    s_clean = re.sub(r"[\n\t\r]+", " ", s_clean).strip()
    words = s_clean.split()
    if not words:
        continue
    first_word = words[0].strip("`'\"([{")
    if first_word in preps:
        prep_violations.append((first_word, s_clean[:70]))
    if len(words) > 25:
        long_sentences.append((len(words), s_clean[:80]))

print(f"Preposition opener violations: {len(prep_violations)}")
for p, s in prep_violations:
    print(f"  {p}: {s}")

print(f"Sentences > 25 words: {len(long_sentences)}")
for l, s in long_sentences:
    print(f"  ({l} words): {s}")
