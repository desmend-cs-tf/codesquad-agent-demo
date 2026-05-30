import  { DynamicTool } from '@langchain/core/tools';

export const getLessonTopicsTool = new DynamicTool({
    name: "get_lesson_topics",
    description: 
        "Returns the list of topics covered in today's AI Agents lesson. " +
        "Always call this FIRST before posting anything in Slack. " +
        "No input needed. ",
    func: async() => {
        return `
          Today's topics for class were:
          1. Generative AI vs AI Agents
          2. The ReAct loop (Reason, Act, Observe)
          3. Goals vs Workflows
          4. How tool descriptions are used in the prompt
          5. Tokens and cost management
          6. Memory and context window
          7. LangChain setup
        `;
    }
});

export const postToSlackTool = new DynamicTool({
    name: "post_to_slack",
    description:
        "Posts a message to the CodeSquad class random channel. " +
        "Use this AFTER you have context ready to share. " +
        "Input: the message text as a plain string. Keep it short, simple, clear and concise.",
    func: async(message) => {
        const webhookUrl = process.env.SLACK_WEBHOOK_URL;

        if(!webhookUrl) {
            return "Error: You do not have the correct SLACK Webhook URL"
        }

        const response = await fetch(webhookUrl, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: message }),
        })

        if(response.ok) {
            return "Messaged posted to Slack successful!"
        } else {
            return "Slack message has failed!!"
        }
    }
})