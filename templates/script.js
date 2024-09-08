// Grabbing the required elements
const newChatButton = document.getElementById('new-chat-button');
const chatList = document.getElementById('chat-list');
const chatTitle = document.getElementById('current-chat-title');

// Function to add a new chat
newChatButton.addEventListener('click', () => {
    // Prompt user for the new chat name
    const newChatName = prompt('Enter new chat topic:');

    if (newChatName) {
        // Create a new chat item in the sidebar
        const newChatItem = document.createElement('li');
        newChatItem.classList.add('chat-item');
        newChatItem.textContent = newChatName;
        chatList.appendChild(newChatItem);

        // Set new chat as the active chat and update the chat title
        chatTitle.textContent = newChatName;
        
        // Scroll to bottom of chat list if it overflows
        chatList.scrollTop = chatList.scrollHeight;
    }
});
