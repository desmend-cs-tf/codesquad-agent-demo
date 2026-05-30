import { tool }   from "@langchain/core/tools";
import { z }      from "zod";
import { Resend } from "resend";

const resend = new Resend(process.env.RESEND_API_KEY);

export const sendEmailDigestTool = tool(
  async ({ subject, body }) => {
    try {
      const { data, error } = await resend.emails.send({
        from:    "CodeSquad Digest <onboarding@resend.dev>",
        to:      [process.env.DIGEST_EMAIL],
        subject: subject,
        text:    body,
      });

      if (error) return `Email failed: ${error.message}`;
      return `Email sent to ${process.env.DIGEST_EMAIL}. Message ID: ${data.id}`;

    } catch (err) {
      return `Email error: ${err.message}`;
    }
  },
  {
    name: "send_email_digest",
    description:
      "Sends the tech digest as an email to the CodeSquad instructor. " +
      "Call this AFTER posting to Discord if the goal mentions email. " +
      "The email can be longer and more detailed than the Discord message — " +
      "include full summaries and all URLs. " +
      "Input: a subject line and the full email body as plain text.",
    schema: z.object({
      subject: z.string().describe(
        "Email subject line. Example: 'CodeSquad Tech Digest — May 27'"
      ),
      body: z.string().describe(
        "Full email body with story titles, URLs, and summaries. " +
        "Use newlines for formatting. Can be longer than the Discord message."
      ),
    }),
  }
);
