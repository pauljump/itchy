# Usual Cold-Start Interview

Use this question bank when the user does not have transcript depth yet. Run it conversationally: ask one question at a time, let the user answer fully, then move to the next. Convert each answer into a judgment entry using the schema in `schema/judgment-entry.schema.json` with `provenance.file = "interview"` and `provenance.quote` = the user's answer verbatim (or as close to verbatim as possible).

**Model for this task: session model (the user's own Claude)**

Each question is tagged with a target domain. Aim for 20-30 questions in a single session to build a meaningful corpus. You do not need to ask all questions — stop when the user's answers become repetitive or when you have enough entries across all domains.

---

## Domain: product (10 questions)

**P1.** You have two versions of a feature: one ships in two days and is rough around the edges, the other ships in two weeks and is clean. Which do you ship, and why?

**P2.** You have a product that a small group of people love and a large group finds confusing. Do you fix the confusion or protect the love? What's your reasoning?

**P3.** A feature is technically impressive but adds complexity to the interface. The simpler alternative is less impressive but users understand it immediately. Which do you build?

**P4.** Your product does two things. One of them is clearly better than the other, and users only come for that one. Do you cut the weak half or keep both? What makes you decide?

**P5.** You discover that the thing you built is being used in a way you didn't intend — and the unintended use is more compelling than the original. Do you follow the users or stick to the original vision? Why?

**P6.** You have a live product with real users and a new product idea you're excited about. How do you decide which gets your attention this week?

**P7.** Someone asks you to add a feature that makes perfect sense for them but doesn't fit your product's direction. How do you handle it?

**P8.** You have to name a product or feature right now. What's your process? Walk me through a real example if you can.

**P9.** A competitor launches something almost identical to what you're building. What do you do?

**P10.** You've been building something for three months and it's not working. At what point do you kill it versus keep going, and what does that decision look like?

---

## Domain: design (8 questions)

**D1.** You're designing something and it looks clean but feels cold. How do you make it feel warm without cluttering it?

**D2.** A design you made looks right to you but wrong to the people you're building for. How do you find out which instinct to trust?

**D3.** You have to choose between a design that's visually impressive and one that's immediately understandable. Which do you pick and when?

**D4.** Someone gives you feedback that a design "doesn't feel right" but can't say why. How do you respond?

**D5.** You're iterating on a design and you've done 10 versions. How do you know when to stop?

**D6.** When does a design need to explain itself versus just work? Give me an example of each from your experience.

**D7.** You're building something for a user who is stressed or overwhelmed. How does that change the design?

**D8.** You have to cut half the screen. What stays and what goes?

---

## Domain: factory (8 questions)

**F1.** You're choosing between building something quickly in a way that creates future debt and building it correctly in a way that slows you down now. How do you decide?

**F2.** You discover that two projects you're running share 80% of the same logic. Do you abstract it out or keep them separate? What tips the decision?

**F3.** You have a system that works but that you don't fully understand. Do you document it, replace it, or leave it? What's your threshold?

**F4.** Something breaks in production. Walk me through your instinct for the first 10 minutes.

**F5.** You're setting up a new tool or process. How do you decide whether it goes into your standard setup or stays one-off?

**F6.** You're handing off a project or system to your future self (or someone else). What do you always make sure to include?

**F7.** You have two infrastructure options: one is simpler but has limits, one is more powerful but adds complexity. How do you choose?

**F8.** Something in your stack is annoying to work with but functional. At what point does the annoyance become worth fixing?

---

## Domain: money (5 questions)

**M1.** You have to decide what to charge for something. Walk me through how you get to a number.

**M2.** You could charge more and lose some users or charge less and keep them all. How do you think through that tradeoff?

**M3.** Someone offers to pay you to build something you wouldn't build otherwise. How close does the offer have to be to your values to take it?

**M4.** You have a budget and two things to spend it on. One is more urgent, one is more important. How do you split it?

**M5.** You're deciding whether to invest time in something speculative. What does your calculus look like?

---

## Domain: voice (5 questions)

**V1.** You write something and then read it back and it doesn't sound like you. How do you find your way back?

**V2.** You have to talk about something you made without sounding like you're selling it. How do you do that?

**V3.** You're writing for two audiences at once — one technical, one not. How do you handle it?

**V4.** Someone misreads the tone of something you wrote. How do you decide whether to clarify or let it stand?

**V5.** You have something important to say and 100 words to say it. What gets cut?

---

## Domain: people (4 questions)

**Pe1.** You and someone you're working with disagree about direction. How do you decide who's right, and what do you do if you can't resolve it?

**Pe2.** Someone you trust gives you feedback you think is wrong. How do you respond?

**Pe3.** You're working with someone slower than you on something time-sensitive. How do you handle it?

**Pe4.** You're about to give someone hard feedback. How do you decide what to say and how to say it?

---

## Indirect block — oblique questions (7 questions)

These questions surface values and direction through the side door. They are intentionally harder to answer directly. Let the user sit with them. The answer itself is the data.

**I1.** You make something. It works exactly the way you designed it but does something you'd never have thought to do. Is it more yours, or less?

**I2.** You start a project to solve your own problem. By the time you ship it, you realize the problem it actually solves is someone else's. Does that make it more valuable or less? Why?

**I3.** Something you built is now used every day by someone who doesn't know you made it. Does that feel like success, or does something feel missing?

**I4.** You have two projects: one where you fully understand every piece, one where the output surprises you. Which one do you trust more? Which one do you care about more?

**I5.** You've been working in a direction for a while. Someone shows you a version of the same direction that's much better than yours. Do you feel like you won or lost?

**I6.** You built the right thing but in the wrong way. Someone else built the wrong thing in the right way. Which outcome bothers you more?

**I7.** You make something for someone you know well. They use it exactly as intended but for reasons you didn't expect. Are you satisfied?

---

## Instructions for the interviewer (Claude)

1. Ask one question at a time. Let the user finish before moving on.
2. If the user gives a short answer, probe once: "Can you say more about why?" or "What would the other choice have cost you?" Do not probe more than once per question.
3. After each answer, silently draft the judgment entry. If the answer is too short or too vague to anchor a real judgment, skip the entry — do not force it.
4. Keep `provenance.quote` as verbatim as possible. This is the audit trail.
5. Use `confidence` honestly: a full, specific answer with clear reasoning is 0.9+. A vague or hedged answer is 0.6-0.7. A one-word answer is 0.5 or below.
6. After 20-30 questions, or when the user signals they are done, write all collected entries as JSONL and run redaction before storing them.
7. Domain balance target: try to get at least 3 entries from each of product, factory, and design. The other domains are secondary.
