# MAHAMAIA Systems — Outreach Templates
## The Five Most Important Emails You Will Send

**Document version:** 1.0  
**Date:** 2026-05-16  
**Use:** Copy, personalize, send. Never send a template without personalizing lines marked [PERSONALIZE].

---

## Preamble: How to Use These Templates

Each template has three layers:
- **The subject line** — the only thing that determines whether the email gets opened
- **The body** — written to be read in under 60 seconds on a phone
- **The ask** — one specific, low-friction action

The single most common mistake in outreach is asking for too much in the first email. None of these templates ask for a commitment. They ask for a reply, a 20-minute call, or a yes/no. The commitment comes in the second conversation.

A note on personalization: the lines marked [PERSONALIZE] are not optional. A template sent without personalization is detectable and will be ignored. The personalization signals that you read their work, which is the minimum requirement for any of these relationships to form.

---

## Template 1 — Design Partner LOI Outreach

**Target:** VP Engineering, CTO, or Director of Product at a VMS vendor, smart city platform, or enterprise surveillance operator. Not the CEO. Not a sales contact. The person who owns the technical stack.

**Goal of this email:** A 20-minute call. Not an LOI. Not a demo. A call.

**When to send:** After identifying 10 target organizations and researching their specific infrastructure (what codec they use, what their analytics pipeline looks like, whether they have published anything about storage or bandwidth costs).

---

**Subject:** Surveillance codec built for machine perception — 62% storage reduction at matched detection accuracy

**Body:**

Hi [Name],

[PERSONALIZE: One sentence showing you know their product — "I've been following [Company]'s work on [specific feature/deployment/case study] — the [specific thing] caught my attention."]

I'm building a video codec at MAHAMAIA Systems designed specifically for machine perception pipelines — the use case where the consumer of the video is a GPU running object detection, not a human reviewing footage.

The short version of what we've built: a compression system that represents surveillance frames as semantic latent grids rather than pixels, achieving a 62% bitrate reduction at matched detection accuracy versus H.265. The latent representation can be consumed directly by downstream models without full pixel decoding.

I think this is relevant to [Company] specifically because [PERSONALIZE: one sentence connecting to their specific problem — storage costs, analytics latency, compliance requirements, bandwidth constraints].

I'm looking for one or two design partners to co-evaluate the system on real deployment data — not a sales relationship, a technical collaboration. In exchange: early access to the full system, influence over the evaluation framework we're building, and co-authorship on any resulting publication if the collaboration produces results worth publishing.

Would a 20-minute call make sense? I can work around your schedule.

Preetam Mukherjee  
MAHAMAIA Systems  
preetam@mahamaia.com  
github.com/MAHAMAIA/lewm-vc

---

**Follow-up (send if no reply after 7 days):**

**Subject:** Re: Surveillance codec — machine perception

Hi [Name],

Following up on the note below in case it got buried.

One specific question: is storage cost or analytics latency the bigger constraint in your current pipeline? That shapes which part of what we've built is most relevant to you.

Happy to send the paper and benchmarks first if that's easier than a call.

Preetam

---

**Notes on this template:**

The subject line leads with a specific number (62%) and a specific claim (matched detection accuracy). This filters for the right reader — someone who cares about detection accuracy, not someone who cares about pixel quality.

The "not a sales relationship, a technical collaboration" line is important and must be true. Do not send this email if you are going to pitch them a product in the call. The first call is purely technical. The commercial conversation follows after they have seen the system work on their data.

The co-authorship offer is a genuine differentiator. Most vendors have never been offered co-authorship by a codec company. It signals that the relationship is about building something together, not extracting value.

---

## Template 2 — Co-Author Recruitment for LeWM-Eval Benchmark Paper

**Target:** Authors of papers on machine-oriented compression, VCM, or learned video codecs. Specifically: first or second authors of papers published in the last 18 months on task-driven compression, feature-based coding, or VCM evaluation.

**Goal of this email:** Agreement to run their codec on LeWM-Eval and contribute a result to the benchmark paper in exchange for co-authorship.

**When to send:** After LeWM-Eval PyPI package exists and at least one codec other than LeWM-VC has been evaluated. You need to be able to say "the evaluation takes less than two hours to run" with evidence.

---

**Subject:** Co-authorship on LeWM-Eval benchmark paper — running your codec takes <2 hours

**Body:**

Hi [Name],

[PERSONALIZE: One sentence referencing their specific paper — "Your work on [paper title] — specifically the [specific finding or method] — is directly relevant to what I'm building."]

I'm the author of LeWM-VC (arXiv 2026.XXXXX) and building LeWM-Eval — a reproducible evaluation framework for machine-oriented video compression. The core idea: measure task accuracy at matched bitrate rather than pixel fidelity, using identical probe architectures across codecs to ensure fair comparison.

I'm assembling the benchmark paper for NeurIPS 2027 Datasets & Benchmarks and would like to include [their codec] in the evaluation. Running LeWM-Eval on a new codec takes under two hours with our CLI: `pip install lewm-eval && lewm-eval run --codec your_codec`. The result goes into the paper.

In exchange: co-authorship on the benchmark paper. The paper's contribution is the framework and the multi-codec comparison — not any individual codec's performance. Your result makes the comparison more comprehensive and more credible.

[PERSONALIZE: One sentence on why their codec specifically is interesting to include — "Your approach to [specific aspect] would stress-test the evaluation in a way that other codecs don't."]

Happy to share the methodology documentation. Is this something worth a quick conversation?

Preetam Mukherjee  
MAHAMAIA Systems  
preetam@mahamaia.com  
[Link to LeWM-Eval repo]  
[Link to arXiv paper]

---

**Follow-up (send if no reply after 10 days):**

**Subject:** Re: LeWM-Eval benchmark paper

Hi [Name],

Quick follow-up — I want to make sure this landed in the right place and didn't get lost.

The specific question: would you be willing to run LeWM-Eval on [their codec] and share the results for inclusion in the benchmark paper? Co-authorship included. The evaluation is fully automated and takes under two hours.

If co-authorship doesn't fit your situation, I'm happy to include the result with acknowledgement only — whatever works for you.

Preetam

---

**Notes on this template:**

The subject line's "takes <2 hours" is the most important phrase. The barrier to participation is friction, not interest. Remove the friction barrier in the subject line.

Do not email the PI/professor first. Email the first or second author of the specific paper. They did the work, they know the codebase, and they have more flexibility. The PI may need to approve, but the first author is the one who will actually run the evaluation.

Send to a maximum of 3 people per group. More than that looks like a mass email and will get ignored.

Target at minimum 15 groups to get 5 confirmed participants. Expect a 30–40% response rate from cold outreach to researchers whose work is directly cited.

---

## Template 3 — Investor Warm Introduction Request

**Target:** A person in your network (AMD contact, academic collaborator, conference acquaintance) who knows a specific investor you want to reach. Never send this to someone who does not actually know the investor.

**Goal of this email:** A warm introduction email they can forward. You are writing the email they will send, not the email you will send.

**When to send:** After you have identified specific investors to target and confirmed that your contact actually knows them (LinkedIn connection alone does not count — they need to have had a real interaction).

---

**Subject:** Introduction request — [Investor Name] at [Fund Name]

**Body:**

Hi [Contact Name],

I'm preparing to raise a seed round for MAHAMAIA Systems and I noticed you're connected to [Investor Name] at [Fund Name].

[PERSONALIZE: One sentence on why this investor specifically — "Their investment in [portfolio company] suggests they understand the deep tech infrastructure space" or "Their writing on [specific post/talk] shows they think about codec and compression infrastructure."]

I'd be grateful for an introduction if you think it's appropriate. To make it easy, here's a note you could forward or adapt:

---

*"[Investor Name] — wanted to introduce you to Preetam Mukherjee, founder of MAHAMAIA Systems. He's building LeWM-VC, a machine-native video codec for surveillance infrastructure — the compression layer for surveillance and edge perception pipelines where the consumer of the video is a GPU, not a human.*

*The technical work is real (public code, pretrained checkpoints, arXiv preprint) and the timing is interesting — MPEG is formalizing Video Coding for Machines right now, which creates a standards play alongside the codec. I thought of you specifically because [PERSONALIZE: reason relevant to the investor's thesis].*

*Worth 20 minutes if you have bandwidth — preetam@mahamaia.com."*

---

No pressure at all if this isn't a fit. And happy to tell you more about what we're building if that would help you decide.

Preetam

---

**Notes on this template:**

Writing the introduction email for your contact is the most important thing this template does. Most people who are willing to make an introduction don't do it because writing the email is work. Remove that friction entirely. Make the forwarded email so clean and accurate that your contact can send it with zero editing.

The forwarded note must not sound like it was written by you. Write it from your contact's voice. This means using "He's building" not "I'm building" and referencing the specific reason the contact thought of this investor.

Never use this template with someone who barely knows you. A weak warm introduction is worse than a cold email — it signals that you don't have real relationships in the space.

---

## Template 4 — MPEG VCM Contribution Cover Note

**Target:** The MPEG VCM working group chair or the contact listed in the current call for contributions (ISO/IEC JTC1/SC29/WG2 or successor). Also useful for the conference version — a position paper submission to a VCM workshop.

**Goal:** Getting a contribution document accepted for discussion at an MPEG plenary, or accepted to a VCM workshop. This is not a sales email — it is a standards process engagement.

**When to send:** After the LeWM-Eval package is on PyPI and at least one external codec has been evaluated. You need to be able to say "operational" not "planned."

---

**Subject:** Contribution: Reproducible Semantic Evaluation Framework for Video Coding for Machines

**Body:**

Dear [Name / Working Group],

I am writing to submit a contribution for consideration at [meeting number/workshop name]: a reproducible evaluation framework for machine-oriented video compression, designated LeWM-Eval.

The contribution addresses a gap in current VCM evaluation practice. Existing proposals measure task accuracy using codec-specific feature extractors or proprietary evaluation pipelines, making cross-codec comparison unreliable. LeWM-Eval proposes a standardized methodology based on: (1) bitrate matching within ±5% BPP, (2) identical lightweight probe architectures trained against frozen teacher detector pseudo-labels, and (3) rate-distortion-accuracy curves as the primary metric rather than point estimates.

Key properties of the framework:

— Codec-agnostic: accepts any codec through a standard encode/decode interface  
— Reproducible: fixed random seeds, locked train/test splits, public checkpoint hashes  
— Open source: available at [repo URL], installable via pip  
— Multi-codec results: [N] codecs evaluated on PEViD-HD surveillance benchmark with consistent methodology

The accompanying paper (arXiv:2026.XXXXX) provides detailed methodology, baseline results across [N] codecs, and analysis of cases where task accuracy rankings diverge from PSNR rankings — a finding directly relevant to VCM evaluation design.

I believe LeWM-Eval could complement the VCM evaluation framework as a reference implementation for semantic preservation metrics. I welcome feedback from the working group on alignment with existing VCM metric definitions and would be glad to adapt terminology or methodology to reduce integration friction.

[PERSONALIZE: If you have attended a previous MPEG meeting or know someone in the working group — "I had the opportunity to discuss related work with [Name] at [event], who suggested submitting this contribution."]

I am available to present this contribution at [meeting location/date] if that would be useful.

Preetam Mukherjee  
MAHAMAIA Systems  
preetam@mahamaia.com

---

**Notes on this template:**

MPEG contributions are formal documents with a specific format (contribution number, source, title, purpose, content). The email above is the cover note — the actual contribution must follow MPEG formatting conventions. Download a recent contribution from the MPEG document server to understand the format before submitting.

The phrase "I welcome feedback from the working group on alignment with existing VCM metric definitions" is important. It signals that you are not arriving with a finished product to impose — you are offering a contribution and inviting collaboration. MPEG participants respond poorly to external actors who appear to be seeking endorsement without engagement.

Observer registration for MPEG plenary meetings is possible but varies by meeting. Check the current policy at the time of submission.

---

## Template 5 — Co-Founder Recruiting

**Target:** A specific person — not a job posting response — who you have identified as a potential co-founder based on their public work (GitHub, papers, LinkedIn, conference talks). Two profiles: production engineering (video/codec infrastructure) or enterprise BD (surveillance/security).

**Goal:** A conversation. Not a job offer. Not a pitch. A conversation about what they are working on and whether there is a fit worth exploring.

**When to send:** Immediately. This is the highest-leverage relationship to build and the one with the longest lead time.

---

**Subject (engineering profile):** Codec infrastructure role — building the machine-oriented compression stack

**Subject (BD profile):** Founding team role — surveillance infrastructure company, seed stage

**Body (engineering profile):**

Hi [Name],

[PERSONALIZE: One specific sentence about their work — "Your [specific project/paper/talk] on [specific topic] — particularly [specific thing] — is exactly the problem I've been thinking about from the codec side."]

I'm the founder of MAHAMAIA Systems. We're building LeWM-VC, a machine-native video codec for surveillance infrastructure — the compression layer for edge AI and smart city deployments where the consumer of the video is a GPU, not a human.

The technical foundation is real: working codec (LeWM-VC, public on GitHub), a 62% bitrate reduction at matched detection accuracy, and 80+ fps on a T4 GPU. The next phase is the part that needs a co-founder with production infrastructure depth: taking a research codec to something that can be embedded in a Jetson, certified by a camera OEM, and deployed in a VMS pipeline.

[PERSONALIZE: One sentence on why them specifically — "Your work on [specific thing] suggests you've solved exactly the [specific problem] we hit at the edge deployment stage."]

I'm not pitching a job. I'm looking for a technical co-founder — someone who would shape the architecture and own the production engineering track. Equity, not salary, for the right person at this stage.

Would a conversation make sense? Happy to share the codebase and the paper first if that's useful.

Preetam Mukherjee  
preetam@mahamaia.com  
github.com/MAHAMAIA/lewm-vc

---

**Body (BD/enterprise profile):**

Hi [Name],

[PERSONALIZE: One specific sentence — "Your work at [Company] on [specific deal or product launch] — specifically the [specific thing] — is exactly the go-to-market challenge I'm thinking through for what we're building."]

I'm the founder of MAHAMAIA Systems. We're building LeWM-VC, a machine-native video codec for surveillance and machine perception infrastructure — the compression layer where the consumer of the video is a GPU running object detection, not a human reviewing footage.

The technology is real and working (public code, arXiv preprint). The next phase is the part that needs a co-founder with enterprise experience in the surveillance and physical security space: finding and closing the first design partners, building the relationships with VMS vendors and camera OEMs, and translating a technical product into a procurement conversation.

[PERSONALIZE: One sentence — "Your background at [Company] with [specific type of customer] suggests you've navigated exactly the procurement cycle we're targeting."]

I'm not looking to hire someone into a BD role. I'm looking for a co-founder who would own the commercial track and shape the company. The right person would have a view on which market to enter first, which customers to prioritize, and how to position this to a CTO at a municipality versus a VP Engineering at a camera OEM.

Would a conversation make sense?

Preetam Mukherjee  
preetam@mahamaia.com

---

**Follow-up (send after 10 days if no reply):**

**Subject:** Re: [original subject]

Hi [Name],

Following up in case this got lost.

One question, no strings: in your current work, is the biggest constraint on the machine perception side the storage cost, the analytics latency, or the compliance/privacy requirements? I'm trying to understand which of those is the dominant pain point in the market you know best.

Happy to share the paper if that gives useful context.

Preetam

---

**Notes on this template:**

"I'm not pitching a job" is the most important line in the co-founder template. Engineers who are good enough to be co-founders are receiving job offers constantly. The co-founder conversation is fundamentally different — it is about ownership, direction, and building something together. That distinction must be clear in the first email or the conversation will default to a recruiting dynamic.

The follow-up question is designed to be answerable without a call — it signals genuine curiosity about their perspective, not just interest in their skills.

For co-founder outreach, expect to contact 20–30 people to have 5–8 serious conversations to find 1–2 who are genuinely interested. This is a numbers game with a high bar. Start immediately and run it in parallel with everything else.

---

## General Outreach Principles

**Volume with quality.** These templates require personalization. A personalized email to 10 people outperforms a generic email to 100 people every time for the relationships that matter.

**Follow-up is not optional.** First emails get a 20–30% response rate from cold outreach to relevant targets. Follow-ups get another 15–20% of the remainder. Send exactly one follow-up, 7–10 days after the first email. Never send a third cold email.

**Track everything.** Maintain a simple spreadsheet: contact name, organization, email sent date, follow-up date, reply status, next action. The outreach process has enough moving parts that memory is not a reliable tracking system.

**Reply speed matters.** When someone replies to any of these emails, respond within 24 hours. Response speed signals seriousness and respect. A 3-day reply to an interested investor or co-founder candidate is a costly signal.

**The goal of every email is one thing.** Not to close a deal. Not to explain the full thesis. To get a reply that leads to a conversation. Every word in these templates should be evaluated against that goal.

---

*Internal document — MAHAMAIA Systems — 2026-05-16*
