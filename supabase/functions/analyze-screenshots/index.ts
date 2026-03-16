import "jsr:@supabase/functions-js/edge-runtime.d.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Client-Info, Apikey",
};

const SYSTEM_PROMPT = `You are an expert sports betting analyst. The user will send you one or more FanDuel sportsbook screenshots.

Extract EVERY visible player prop bet and return ONLY a valid JSON array — no markdown, no explanation, nothing else.

Each object must have exactly these fields:
{
  "player":    "Full player name",
  "prop":      "e.g. 20+ Points",
  "odds":      integer (American odds, e.g. -164 or +122),
  "game":      "Team A vs Team B",
  "tier":      "LOCK" | "STRONG" | "VALUE",
  "trueProb":  integer 0-100,
  "ev":        "+X.X%" or "-X.X%",
  "kelly":     "$XX"
}

Tier rules (after devigging):

- LOCK   = true probability >= 73%
- STRONG = true probability 60-72%
- VALUE  = true probability < 60% but EV is positive

EV formula: (trueProb/100 x decimalOdds) - 1, expressed as a percentage.
Kelly suggested bet assumes $1,000 bankroll, 25% fractional Kelly.
Return ONLY the JSON array.`;

interface ImageContent {
  type: "image";
  source: {
    type: "base64";
    media_type: string;
    data: string;
  };
}

interface TextContent {
  type: "text";
  text: string;
}

type ContentItem = ImageContent | TextContent;

interface RequestPayload {
  images: Array<{
    data: string;
    mediaType: string;
  }>;
}

Deno.serve(async (req: Request) => {
  if (req.method === "OPTIONS") {
    return new Response(null, {
      status: 200,
      headers: corsHeaders,
    });
  }

  try {
    const apiKey = Deno.env.get("ANTHROPIC_API_KEY");
    if (!apiKey) {
      return new Response(
        JSON.stringify({ error: "API key not configured" }),
        {
          status: 500,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        }
      );
    }

    const payload: RequestPayload = await req.json();

    if (!payload.images || !Array.isArray(payload.images) || payload.images.length === 0) {
      return new Response(
        JSON.stringify({ error: "No images provided" }),
        {
          status: 400,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        }
      );
    }

    const content: ContentItem[] = [];

    for (const img of payload.images) {
      content.push({
        type: "image",
        source: {
          type: "base64",
          media_type: img.mediaType,
          data: img.data,
        },
      });
    }

    content.push({
      type: "text",
      text: "Extract all player props from these FanDuel screenshots and return the JSON array.",
    });

    const response = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": apiKey,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model: "claude-sonnet-4-20250514",
        max_tokens: 4000,
        system: SYSTEM_PROMPT,
        messages: [{ role: "user", content }],
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      return new Response(
        JSON.stringify({ error: `Anthropic API error: ${response.status}`, details: errorText }),
        {
          status: response.status,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        }
      );
    }

    const data = await response.json();

    return new Response(JSON.stringify(data), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (error) {
    return new Response(
      JSON.stringify({ error: error instanceof Error ? error.message : "Unknown error" }),
      {
        status: 500,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      }
    );
  }
});
