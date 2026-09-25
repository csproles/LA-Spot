using Bunit;
using MelanomaDetection.Web.Components.Chat;
using MelanomaDetection.Web.Models;

namespace MelanomaDetection.Web.Tests;

public sealed class ChatSuggestionsTests : BunitContext
{
    [Fact]
    public void AnEmptyChatOffersTheStarterQuestionsAndSendsTheOneTapped()
    {
        string? sent = null;
        var cut = Render<ChatMessageList>(p => p
            .Add(x => x.Messages, Array.Empty<ChatMessage>())
            .Add(x => x.OnSuggestion, (string question) => sent = question));

        var buttons = cut.FindAll("button.chat-suggestion");
        Assert.Equal(ChatSuggestions.Questions.Length, buttons.Count);
        Assert.Contains(buttons, b => b.TextContent.Trim() == "What is melanoma?");
        Assert.Contains(buttons, b => b.TextContent.Trim() == "What type of doctor should I see about a mole?");

        buttons.Single(b => b.TextContent.Trim() == "What does the asymmetry score mean for melanoma?").Click();

        Assert.Equal("What does the asymmetry score mean for melanoma?", sent);
    }

    [Fact]
    public void TheSuggestionsGoAwayOnceTheConversationStarts()
    {
        var cut = Render<ChatMessageList>(p => p
            .Add(x => x.Messages, [new ChatMessage { Role = ChatRole.User, Text = "What is melanoma?" }])
            .Add(x => x.OnSuggestion, (string _) => { }));

        Assert.Empty(cut.FindAll("button.chat-suggestion"));
    }

    [Fact]
    public void WhileTheFirstAnswerIsLoadingNoSuggestionsShow()
    {
        var cut = Render<ChatMessageList>(p => p
            .Add(x => x.Messages, Array.Empty<ChatMessage>())
            .Add(x => x.IsSending, true)
            .Add(x => x.OnSuggestion, (string _) => { }));

        Assert.Empty(cut.FindAll("button.chat-suggestion"));
    }
}
