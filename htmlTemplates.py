css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #edeff2
}
.chat-message.bot {
    background-color: #edeff2
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: black;
}
'''

import base64

# Read the image file
with open("images/chatbot.png", "rb") as image_file:
    # Encode the image into a Base64 string
    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRezya9NlI1A8O0SmqGiYI6RcHE8chOZGGwtWxbZzP9xA&s" style="max-height: 55px; max-width: 36px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://freesvg.org/img/abstract-user-flat-1.png" style="max-height: 55px; max-width: 36px; border-radius: 50%; object-fit: cover;">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''

