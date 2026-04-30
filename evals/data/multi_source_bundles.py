"""
Source bundles for the multi-source competitive intelligence brief pipeline.

Each target company has five entries:
  - overview:    high-level company description, 2-3 paragraphs.
                 Used by the scoping agent (Group 1) to extract a context block.
  - competitive: competitor landscape, market position, market share signals.
  - product:     product portfolio, technology stack, recent launches.
  - financial:   funding history, valuation, revenue, hiring signals.
  - risk:        legal, regulatory, compliance, controversies.

Bundles are hand-curated from public sources (company websites, Wikipedia,
mainstream press coverage) as of late 2024 / early 2025. They are deliberately
*fixed* — this pipeline tests execution strategies, not retrieval. Both
Agentic Capsules and the LangGraph baseline see identical input.

Token budget: ~150-250 words per bundle, sized so the full per-arm payload
(bundle + extraction instruction) fits comfortably in a single LLM call on
all three Track A models (haiku, sonnet, gpt-4o-mini).
"""

TARGETS: dict[str, dict[str, str]] = {
    # ------------------------------------------------------------------
    # Stripe — payments infrastructure, private, founded 2010
    # ------------------------------------------------------------------
    "Stripe": {
        "overview": (
            "Stripe is a financial infrastructure company founded in 2010 by Irish "
            "brothers Patrick and John Collison. Headquartered in Dublin and South "
            "San Francisco, the company provides payment processing APIs and a "
            "broader suite of financial tools used by businesses ranging from "
            "small startups to large enterprises. Stripe's core thesis is that "
            "the internet's payments and financial infrastructure should be as "
            "programmable as compute and storage. The company remains private as "
            "of 2025 and has been the subject of recurring IPO speculation since "
            "2021. Stripe processes hundreds of billions of dollars in payment "
            "volume annually across more than forty countries."
        ),
        "competitive": (
            "Stripe's primary competitors in card payments include Adyen "
            "(Amsterdam-based, public), PayPal Holdings, and Block Inc.'s Square "
            "subsidiary. Adyen is the closest peer in enterprise card "
            "processing, particularly in Europe and for omnichannel retailers; "
            "PayPal owns Braintree, which competes directly for developer-first "
            "online merchants. Block dominates US small-business in-person "
            "payments. In emerging markets Stripe faces local champions such as "
            "Razorpay in India and dLocal in Latin America. Stripe lost Shopify "
            "as an exclusive partner in 2017 when Shopify launched Shop Pay, "
            "though Stripe remains Shopify's primary card processor. Stripe's "
            "differentiation has historically been developer experience: clean "
            "APIs, extensive documentation, and a fast onboarding flow. "
            "Competitors have closed the API gap meaningfully since 2020."
        ),
        "product": (
            "Stripe's product portfolio includes Stripe Payments (the original "
            "card-processing API), Stripe Connect (multi-party marketplace "
            "payouts), Stripe Atlas (US company incorporation for foreign "
            "founders), Stripe Radar (machine-learning fraud prevention), Stripe "
            "Capital (merchant cash advances), Stripe Issuing (programmable "
            "card issuance), Stripe Treasury (banking-as-a-service in "
            "partnership with Evolve Bank), Stripe Tax (sales-tax automation), "
            "Stripe Climate (carbon removal funding), and Stripe Terminal (in-"
            "person card readers). The company has invested heavily in AI-"
            "assisted features in 2024, including Radar fraud detection model "
            "upgrades and developer-tool integrations. Stripe Sigma exposes "
            "transaction data via SQL for analytics. The Stripe CLI and the "
            "Stripe Shell were released to streamline developer testing."
        ),
        "financial": (
            "Stripe was last valued at approximately $70 billion in a February "
            "2024 secondary tender offer, a recovery from a $50 billion "
            "internal markdown in early 2023 and a peak of $95 billion in 2021. "
            "Investors include Sequoia Capital, Andreessen Horowitz, General "
            "Catalyst, Founders Fund, GIC, and Allianz X. Reported total payment "
            "volume crossed $1 trillion in 2023. The company conducted a major "
            "round in March 2023 to provide liquidity for early employees and "
            "to cover RSU-related tax obligations. Headcount peaked above "
            "8,000 in 2022, was reduced by approximately 14% in November 2022 "
            "during the broader tech downturn, and has grown modestly since. "
            "Stripe has not disclosed audited revenue but is widely reported "
            "to be cash-flow positive."
        ),
        "risk": (
            "As a payments company, Stripe is subject to PCI DSS compliance, "
            "card network rules, and country-specific financial regulations "
            "including money-transmitter licensing in US states. Stripe has "
            "faced periodic merchant disputes over account freezes and reserve "
            "holds, particularly affecting high-risk verticals; these "
            "complaints surface regularly on Hacker News and consumer forums "
            "but have not resulted in material legal exposure. The company has "
            "navigated US sanctions enforcement carefully given its "
            "international footprint. Stripe declined to process payments for "
            "certain controversial categories during the Trump administration, "
            "drawing political criticism from both sides at different times. "
            "The 2022 layoffs prompted scrutiny of the company's prior "
            "headcount growth. No major data breach or regulatory enforcement "
            "action has been publicly disclosed."
        ),
    },

    # ------------------------------------------------------------------
    # Anthropic — AI safety, private, founded 2021
    # ------------------------------------------------------------------
    "Anthropic": {
        "overview": (
            "Anthropic is an AI safety company founded in 2021 by Dario "
            "Amodei, Daniela Amodei, and several former senior researchers "
            "from OpenAI. Headquartered in San Francisco, the company focuses "
            "on building reliable, interpretable, and steerable large language "
            "models. Its flagship product is Claude, a family of LLMs offered "
            "via a consumer chat interface, an API for developers, and "
            "integrations such as Claude Code (a CLI coding assistant). "
            "Anthropic has positioned itself as the AI industry's safety-"
            "first commercial lab, publishing research on constitutional AI, "
            "interpretability, and red-teaming in addition to shipping "
            "production models. The company has raised more than $10 billion "
            "since founding and remains private."
        ),
        "competitive": (
            "Anthropic competes most directly with OpenAI, which was founded "
            "by overlapping people and dominates the consumer chatbot category "
            "through ChatGPT. Other major competitors include Google DeepMind "
            "(Gemini), Meta (open-weight Llama), Mistral AI (European open-"
            "weight challenger), xAI (Elon Musk's Grok), and Cohere (enterprise-"
            "focused). On the enterprise API side, Anthropic competes with "
            "OpenAI's API and with hyperscaler-bundled offerings from AWS "
            "Bedrock, Google Vertex AI, and Microsoft Azure OpenAI Service. "
            "Anthropic differentiates on safety research, longer context "
            "windows (200K tokens standard, 1M for some customers), and "
            "developer ergonomics. Independent benchmarks consistently rank "
            "Claude near the top on coding and reasoning tasks. Market share "
            "in API revenue is estimated to be a distant second to OpenAI but "
            "growing faster on a percentage basis."
        ),
        "product": (
            "Anthropic's main product line is the Claude family of large "
            "language models. As of late 2024 the lineup includes Claude Opus "
            "(highest-capability frontier model), Claude Sonnet (balanced "
            "speed and capability), and Claude Haiku (fast and cheap). Claude "
            "is accessed through claude.ai (consumer chat), the Anthropic API, "
            "AWS Bedrock, and Google Vertex AI. Claude Code, released in 2024, "
            "is a terminal-based coding assistant that integrates with the "
            "Claude API. Anthropic has invested in tool-use capabilities, "
            "computer-use agents, and prompt caching for cost reduction. The "
            "company publishes a Model Card and Usage Policies for each major "
            "release. Constitutional AI is the company's signature alignment "
            "technique, in which models are trained to follow a written "
            "constitution rather than purely learning from human feedback."
        ),
        "financial": (
            "Anthropic has raised more than $10 billion in disclosed funding "
            "since 2021. Major investors include Google (multi-billion "
            "commitment), Amazon (committed up to $8 billion), Spark Capital, "
            "Lightspeed Venture Partners, Bessemer, Salesforce Ventures, and "
            "Menlo Ventures. The company was valued at approximately $40 "
            "billion in mid-2024 and reportedly more than $60 billion by late "
            "2024 in subsequent rounds. Annualized revenue reportedly grew "
            "from roughly $100 million at the start of 2024 to more than $800 "
            "million by year-end, driven primarily by API consumption from "
            "developers and enterprise customers. Headcount has grown rapidly "
            "from approximately 300 employees at the start of 2024 to more "
            "than 700 by year-end, with continued aggressive hiring in "
            "research, infrastructure, and go-to-market roles."
        ),
        "risk": (
            "Anthropic operates in a rapidly evolving regulatory environment. "
            "The EU AI Act, finalized in 2024, imposes obligations on "
            "general-purpose AI models above certain compute thresholds and "
            "Anthropic's frontier models will fall within scope. California "
            "passed and then partially vetoed AI safety legislation (SB 1047) "
            "in 2024 that would have imposed direct obligations on frontier "
            "labs. Anthropic publicly supported a modified version of the "
            "bill. The company faces ongoing copyright litigation from "
            "authors and publishers regarding training data, similar to "
            "lawsuits filed against OpenAI. Music publishers have separately "
            "sued Anthropic over song-lyric reproduction in Claude outputs. "
            "Anthropic publishes a Responsible Scaling Policy that commits to "
            "specific safety evaluations before deploying more capable models, "
            "and updates this policy publicly when revised."
        ),
    },

    # ------------------------------------------------------------------
    # Figma — collaborative design, private (post-Adobe-deal), founded 2012
    # ------------------------------------------------------------------
    "Figma": {
        "overview": (
            "Figma is a browser-based collaborative design platform founded "
            "in 2012 by Dylan Field and Evan Wallace. Headquartered in San "
            "Francisco, the company built the first widely-adopted design "
            "tool that runs entirely in the web browser, enabling real-time "
            "multiplayer editing similar to Google Docs but for vector "
            "graphics and UI mockups. Figma's design files are accessible via "
            "URL and editable by multiple users simultaneously, which "
            "transformed the design industry's collaboration patterns. The "
            "company was the subject of a high-profile $20 billion "
            "acquisition agreement with Adobe announced in September 2022; "
            "the deal was terminated in December 2023 after the UK Competition "
            "and Markets Authority and the European Commission signaled they "
            "would block it on competition grounds."
        ),
        "competitive": (
            "Figma's primary historical competitor was Adobe XD, which Adobe "
            "effectively wound down following the failed acquisition; Adobe "
            "stopped active XD development in 2023. Other competitors include "
            "Sketch (Mac-only, the original modern UI design tool that Figma "
            "displaced), Framer (also browser-based, with stronger interactive "
            "prototyping and a website-builder pivot), and Penpot (open-"
            "source). InVision exited the design-tool market in 2024. On the "
            "whiteboarding side Figma's FigJam product competes with Miro and "
            "Mural. Figma has near-dominant market share among professional "
            "product-design teams at technology companies; surveys of "
            "designers consistently show Figma usage above 70%. New entrants "
            "since 2023 have focused on AI-assisted design generation, where "
            "Figma is itself rolling out AI features rather than facing "
            "displacement."
        ),
        "product": (
            "Figma's product portfolio includes Figma Design (the core "
            "vector-based UI design tool), FigJam (an online whiteboarding "
            "and brainstorming product), Dev Mode (a developer-focused view "
            "of design files with code snippets and inspect tools, launched "
            "in 2023), Figma Slides (a collaborative presentation product "
            "announced at Config 2024), and Figma AI (a suite of AI-assisted "
            "features including text-to-design generation, asset search, and "
            "auto-rename). The platform supports plugins and widgets developed "
            "by third parties, with a substantial community marketplace. "
            "Figma's rendering engine is built in C++ compiled to WebAssembly "
            "for performance, with a custom-built multiplayer synchronization "
            "system. The company hosts an annual user conference, Config, "
            "which has grown to over 8,000 in-person attendees."
        ),
        "financial": (
            "Figma was last publicly valued at $20 billion based on the "
            "terminated Adobe acquisition agreement of September 2022. As "
            "compensation for the failed deal, Adobe paid Figma a $1 billion "
            "termination fee in December 2023. Prior to the Adobe agreement, "
            "Figma had raised approximately $333 million from investors "
            "including Index Ventures, Greylock, Kleiner Perkins, Sequoia, and "
            "Andreessen Horowitz, with a 2021 round valuing the company at "
            "$10 billion. Reported annual recurring revenue was approximately "
            "$400 million at the time of the Adobe deal announcement and "
            "exceeded $600 million by 2024 according to press reports. The "
            "company has been profitable on a cash-flow basis. Figma began a "
            "secondary tender offer in 2024 to provide employee liquidity. "
            "Headcount is approximately 1,500 globally."
        ),
        "risk": (
            "The Adobe deal collapse in December 2023 was the company's "
            "highest-profile risk event, costing more than fifteen months of "
            "executive attention and uncertainty for employees and "
            "shareholders. The UK CMA and the European Commission both "
            "indicated they would block the deal on competition grounds, "
            "citing the elimination of Figma as a competitor to Adobe XD. "
            "The deal termination was mutual rather than blocked outright, "
            "with Adobe paying the $1 billion break fee. Figma is subject to "
            "GDPR in the EU, CCPA in California, and standard SaaS data-"
            "protection obligations; no major breach has been publicly "
            "disclosed. The launch of Figma AI in mid-2024 was followed by "
            "the rapid removal of one feature, Make Design, after users "
            "discovered it produced outputs strikingly similar to existing "
            "Apple iOS app designs, raising questions about training-data "
            "provenance. Figma paused the feature pending investigation."
        ),
    },
}


# Lens names in canonical order. Used by the pipeline factory to construct
# the four research-arm groups in a stable order across runs.
LENSES: tuple[str, ...] = ("competitive", "product", "financial", "risk")
