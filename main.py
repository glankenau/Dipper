#!/usr/bin/env python
# pylint: disable=C0116,W0613

import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, Update, ReplyKeyboardMarkup, ReplyKeyboardRemove, ChatAction, Chat, ParseMode
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext, CallbackQueryHandler, ConversationHandler
import openai
import os
import speech_recognition as sr
from pydub import AudioSegment
from gtts import gTTS
import re
from enum import Enum
from threading import Timer

telegram_token = os.environ['TELEGRAM_TOKEN']
openai_api_key = os.environ['OPENAI_API_KEY']
if 'WEBHOOK_URL' in os.environ:
    webhook_url = os.environ['WEBHOOK_URL']

# Enable logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

openai.api_key = openai_api_key
openai_model = "gpt-3.5-turbo"

# enum of supported languages
class Language(Enum):
    English = "en"
    Spanish = "es"

class ChatPreferences:
    def __init__(self, chat_id):
        self.chat_id = chat_id
        self.language = Language.Spanish
        self.personality_prompt = ""
    
class ChatLog:
    def __init__(self, chat: Chat):
        self.chat_id = chat.id
        self.preferences = ChatPreferences(self.chat_id)
        self.is_group_chat = chat.type != Chat.PRIVATE
        self.chat_logs = []
        self.wait_timer = None

    def clear_chat_logs(self):
        self.chat_logs.clear()

saved_chats = {}

# enum that represents a role in a chat
class Role(Enum):
    User = "user"
    System = "system"
    Assistant = "assistant"

# Class for storing chat log entries with message ID, username and message
class ChatLogEntry:
    def __init__(self, message_id, role: Role, message):
        self.message_id = message_id
        self.role = role
        self.message = message

def append_to_chat_log(chat_id: int, message_id: int, role: Role, message: str):
    """Append a message to the chat log."""
    log_entry = ChatLogEntry(message_id, role, message.strip())
    saved_chats[chat_id].chat_logs.append(log_entry)

def get_chat_log(chat_id: int) -> str:
    """Get the chat log."""
    return "\n".join([f"{entry.role}: {entry.message}" for entry in saved_chats[chat_id].chat_logs])

def get_chat_response(chat_id: int, username: str, isVoiceMessage: bool = False) -> str:
    """Get the chat response."""
    # logger.debug(f"Current chatlog is:\n{get_chat_log(chat_id)}")
    
    messages = [{"role": entry.role.value, "content": entry.message} for entry in saved_chats[chat_id].chat_logs]
    personality = saved_chats[chat_id].preferences.personality_prompt
    if personality != "":
        messages.append({"role": Role.System.value, "content": personality})

    response = openai.ChatCompletion.create(
        model=openai_model,
        messages = messages
    )
    response = response.choices[0].message.content
    return response

# Convo states
Waiting_for_chat_message, Waiting_for_raw_prompt = range(2)

admin = 1080659616
auth_list = [2112411445, 1250555072]

# enum of auth modes
class AuthMode(Enum):
    Admin_only = "Admin only"
    White_list = "White list"
    Black_list = "Black list"
    Everyone = "Everyone"

current_auth_mode = AuthMode.White_list

# Check if user is authorized, admin is always authorized
def authorized(user) -> bool:
    """Check if the user is authorized."""
    if user.id == admin: return True
    if current_auth_mode == AuthMode.Everyone:
        logger.warning(f"AuthMode is set to Everyone, allowing user {user.first_name} ({user.id}) to use the bot.")
        return True
    if current_auth_mode == AuthMode.Admin_only: return False
    if current_auth_mode == AuthMode.White_list:
        logger.warning(f"AuthMode is set to White list, but user {user.first_name} ({user.id}) is not in the list.")
        return user.id in auth_list
    if current_auth_mode == AuthMode.Black_list: return user.id not in auth_list
    

# Handlers:

def switch_auth_mode(update: Update, context: CallbackContext):
    """Switch auth mode."""
    user = update.effective_user
    # only the admin can switch auth mode
    if user.id != admin: return
    global current_auth_mode
    # populate auth mode buttons
    auth_mode_buttons = []
    for auth_mode in AuthMode:
        auth_mode_buttons.append([InlineKeyboardButton(auth_mode.value, callback_data=("auth_mode", auth_mode.value))])
    auth_mode_buttons.append([InlineKeyboardButton("Cancel", callback_data=("auth_mode", "cancel_command"))])
    # send auth mode buttons
    reply_markup = InlineKeyboardMarkup(auth_mode_buttons)
    message = "Current auth mode is set to " + current_auth_mode.value + ".\nSelect a new auth mode:"
    update.message.reply_text(message, reply_markup=reply_markup)

def start(update: Update, context: CallbackContext):
    """Send a message when the command /start is issued."""
    user = update.effective_user
    chat = update.effective_chat
    if not authorized(user): return
    context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    logger.info(f"Message: {update.message.text.lower()}")
    if chat.id not in saved_chats:
        saved_chats[chat.id] = ChatLog(chat)

    if len(context.args) > 0:
        greet_prompt = " ".join(context.args)
    elif saved_chats[chat.id].preferences.language == Language.Spanish:
        greet_prompt = f"Eres un chatbot llamado Leonardo. Escribe un mensaje saludando a {update.effective_user.first_name}."
    elif saved_chats[chat.id].preferences.language == Language.English:
        greet_prompt = f"You are chatbot named Leonardo. Write a message greeting {update.effective_user.first_name}."

    response = openai.ChatCompletion.create(
        model=openai_model,
        messages = [{"role": Role.System.value, "content": greet_prompt}]
    )
    response = response.choices[0].message.content
    reply_message = update.message.reply_text(response, quote=False)
    saved_chats[chat.id].clear_chat_logs()
    append_to_chat_log(chat.id, -1, Role.System, greet_prompt)
    append_to_chat_log(chat.id, reply_message.message_id, Role.Assistant, response)
    return Waiting_for_chat_message

def set_personality_prompt(update: Update, context: CallbackContext):
    """Set the personality prompt."""
    user = update.effective_user
    chat = update.effective_chat
    if not authorized(user): return

    if len(context.args) > 0:
        saved_chats[chat.id].preferences.personality_prompt = " ".join(context.args)
        update.message.reply_text("Personality prompt set.")
    else:
        saved_chats[chat.id].preferences.personality_prompt = ""
        update.message.reply_text("Personality prompt cleared.")

    return Waiting_for_chat_message

def set_chat_model(update: Update, context: CallbackContext):
    """Set the chat model."""
    user = update.effective_user
    chat = update.effective_chat
    if not authorized(user): return

    global openai_model

    if len(context.args) > 0:
        openai_model = context.args[0]
        update.message.reply_text(f"Chat model set to {openai_model}.")
    else:
        update.message.reply_text(f"No chat model specified, current chat model is {openai_model}.")

    return Waiting_for_chat_message

def answer_message(update: Update, context: CallbackContext, chat: Chat, user, isEdit: bool = False):
    context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    response = get_chat_response(chat.id, user.first_name)
    # sometimes the response contains invalid markdown, so we need to retry without markdown if it fails
    try:
        reply_message = update.message.reply_text(response, quote=isEdit, parse_mode=ParseMode.MARKDOWN)
    except:
        reply_message = update.message.reply_text(response, quote=isEdit)

    if isEdit:
        append_to_chat_log(chat.id, reply_message.message_id, Role.Assistant, f"[Quoting edited message] {response}")
    else:
        append_to_chat_log(chat.id, reply_message.message_id, Role.Assistant, response)

    logger.info(f"Response: {response}")

def process_message(update: Update, context: CallbackContext):
    """Reply to the user message."""
    user = update.effective_user
    chat = update.effective_chat
    if not authorized(user): return
    print("Update ID: " + str(update.update_id))
    if update.message:
        print("Message ID: " + str(update.message.message_id))
        print("Message Text: " + str(update.message.text))
        message = update.message.text
        logger.info(f"Message: {message}")
        append_to_chat_log(chat.id, update.message.message_id, Role.User, message)

    elif update.edited_message:
        print("Edited Message ID: " + str(update.edited_message.message_id))
        print("Edited Message Text: " + str(update.edited_message.text))
        # find original message in chat log using message ID
        original_message = "[Message not found]"
        new_message = update.edited_message.text
        for n, entry in enumerate(saved_chats[chat.id].chat_logs):
            if entry.message_id == update.edited_message.message_id:
                # edit the message in the chat log
                original_message = entry.message
                saved_chats[chat.id].chat_logs[n].message = f"[Edited] {new_message}"

        # get chat response
        prompt = f"{update.effective_user.first_name} edited a prevous message.\n"
        prompt += f"Original message: {original_message}\n"
        prompt += f"New message: {new_message}\n"
        prompt += f"Say something to acknowledge the edit."
        append_to_chat_log(chat.id, -1, Role.System, prompt)

    else:
        print("Unsupported message type")
        return
    
    wait = 5
    try:
        saved_chats[chat.id].wait_timer.cancel()
    except(AttributeError):
        pass
    saved_chats[chat.id].wait_timer = Timer(wait, answer_message, [update, context, chat, user, update.message == None])
    saved_chats[chat.id].wait_timer.start()

    return Waiting_for_chat_message


def answer_voice_message(update: Update, context: CallbackContext, chat: Chat, user):
    current_language = saved_chats[chat.id].preferences.language
    context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.RECORD_AUDIO)
    response = get_chat_response(chat.id, user.first_name, isVoiceMessage=True)
    tts = gTTS(text=response, lang="es" if current_language == Language.Spanish else "en", slow=False)
    tts.save("reply.mp3")
    # sound = AudioSegment.from_wav("reply.wav")
    # sound.export("reply.mp3", format="mp3")
    reply_message = update.message.reply_voice(voice=open("reply.mp3", "rb"))
    append_to_chat_log(chat.id, reply_message.message_id, Role.Assistant, f"[Voice message transcription]: {response}")
    logger.info(f"Response: {response}")

def process_voice_message(update: Update, context: CallbackContext):
    user = update.effective_user
    chat = update.effective_chat
    if not authorized(user): return
    # get basic info about the voice note file and prepare it for downloading
    new_file = context.bot.get_file(update.message.voice.file_id)
    # download the voice note as a file
    new_file.download("voice_note.ogg")
    sound = AudioSegment.from_ogg("voice_note.ogg")
    sound.export("voice_note.wav", format="wav")
    r = sr.Recognizer()
    with sr.AudioFile("voice_note.wav") as source:
        # listen for the data (load audio to memory)
        audio_data = r.record(source)
        # recognize (convert from speech to text)
        current_language = saved_chats[chat.id].preferences.language
        message = r.recognize_google(audio_data, language="es-MX" if current_language == Language.Spanish else "en-US")
        logger.info(f"Voice message: {message}")
        append_to_chat_log(chat.id, update.message.message_id, Role.User, f"[Voice message transcription]: {message}")

        wait = source.DURATION + 10
        try:
            saved_chats[chat.id].wait_timer.cancel()
        except(AttributeError):
            pass
        saved_chats[chat.id].wait_timer = Timer(wait, answer_voice_message, [update, context, chat, user])
        saved_chats[chat.id].wait_timer.start()

        return Waiting_for_chat_message

def process_raw_prompt(update: Update, context: CallbackContext):
    """Pass the raw prompt to the AI."""
    user = update.effective_user
    if not authorized(user): return
    context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    message = update.message.text
    logger.info(f"Message: {message}")
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=message,
        temperature=0.9,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0.6,
        presence_penalty=0.6
    )
    response = response.choices[0].text
    logger.info(f"Raw response: {response}")
    update.message.reply_text(response)
    return Waiting_for_raw_prompt

def switch_to_raw_prompt(update: Update, context: CallbackContext):
    """Switch to raw prompt."""
    user = update.effective_user
    if not authorized(user): return
    update.message.reply_text("I will now pass your messages directly to OpenAI. Write a message to continue.")
    return Waiting_for_raw_prompt

def switch_to_chat_mode(update: Update, context: CallbackContext):
    """Switch to chat mode.""" 
    user = update.effective_user
    if not authorized(user): return
    update.message.reply_text("Chat mode is now enabled, I try to keep track of our recent conversation.")
    update.message.reply_text("If something goes wrong, you can restart the conversation by typing /start.", quote=False)
    update.message.reply_text("Write a message to continue.", quote=False)
    return Waiting_for_chat_message

def switch_language(update: Update, context: CallbackContext):
    """Switch the language of the chatbot."""
    user = update.effective_user
    chat = update.effective_chat
    if not authorized(user): return
    message = "Switching the language of the chatbot will reset the conversation. Current language: " + saved_chats[chat.id].preferences.language.name
    # populate the buttons with the supported languages in the enum
    supported_language_buttons = [[InlineKeyboardButton(language.name, callback_data=("switch_language", language.value))] for language in Language]
    supported_language_buttons.append([InlineKeyboardButton("Cancel", callback_data=("switch_language", "cancel_command"))])

    reply_markup = InlineKeyboardMarkup(supported_language_buttons)
    update.message.reply_text(message, reply_markup=reply_markup)

def help_command(update: Update, context: CallbackContext):
    """Send a message when the command /help is issued."""
    user = update.effective_user
    if not authorized(user): return
    help_text = """
Inicia (o reinicia) la conversación enviando el comando /start.
Puedes enviar mensajes de texto o de voz y el bot tratará de manetener la conversación.
También puedes editar tus mensajes de texto para corregir errores (Experimental).

Usa /raw_mode para pasar los mensajes directamente a OpenAI.
Usa /chat_mode para cabia el modo de conversación a "chat" (este es el modo por defecto).
Para cambiar el idioma del bot, usa /switch_language.
    """
    update.message.reply_text(help_text)

def process_start_message(update: Update, context: CallbackContext):
    """Reply the user message."""
    user = update.effective_user
    if not authorized(user): return
    logger.info("The user has started the conversation without the /start command.")
    update.message.reply_text('Hi! Send the /start command to start the conversation.')

def handle_language_button_press(update: Update, context: CallbackContext):
    """Parses the CallbackQuery and updates the message text."""
    user = update.effective_user
    chat = update.effective_chat
    if not authorized(user): return
    query = update.callback_query
    # CallbackQueries need to be answered, even if no notification to the user is needed
    # Some clients may have trouble otherwise. See https://core.telegram.org/bots/api#callbackquery
    query.answer()
    current_language = saved_chats[chat.id].preferences.language
    if query.data[1] == "cancel_command":
        query.edit_message_text(text=f"OK, I'll keep speaking {current_language.name}.")
        return
    else:
        current_language = Language(query.data[1])
        saved_chats[chat.id].preferences.language = current_language
        query.edit_message_text(text=f"OK, I'll speak {current_language.name} from now on.")
    return ConversationHandler.END

def handle_auth_mode_button_press(update: Update, context: CallbackContext):
    """Parses the CallbackQuery and updates the message text."""
    user = update.effective_user
    query = update.callback_query
    # only the admin can switch auth mode
    if user.id != admin:
        query.message.reply_text(f"{user.first_name}, you are not authorized to use this command.")
        return
    # CallbackQueries need to be answered, even if no notification to the user is needed
    # Some clients may have trouble otherwise. See https://core.telegram.org/bots/api#callbackquery
    query.answer()
    global current_auth_mode
    if query.data[1] == "cancel_command":
        query.edit_message_text(text=f"OK, I'll keep using {current_auth_mode.value} auth mode.")
        return
    else:
        current_auth_mode = AuthMode(query.data[1])
        query.edit_message_text(text=f"OK, I'll use {current_auth_mode.value} auth mode from now on.")

bot_commands = [
    CommandHandler('chat_mode', switch_to_chat_mode),
    CommandHandler('raw_mode', switch_to_raw_prompt),
    CommandHandler('personality', set_personality_prompt),
    CommandHandler('set_model', set_chat_model),
    CommandHandler('switch_language', switch_language), CallbackQueryHandler(handle_language_button_press, pattern=lambda c: c[0] == "switch_language"),
    CommandHandler('auth_mode', switch_auth_mode), CallbackQueryHandler(handle_auth_mode_button_press, pattern=lambda c: c[0] == "auth_mode")
    ]

def main() -> None:
    """Start the bot."""
    logger.info("Starting bot...")
    
    # Create the Updater and pass it your bot's token.
    updater = Updater(telegram_token, arbitrary_callback_data=True)

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    conv_handler = ConversationHandler(
        entry_points=[MessageHandler(Filters.text & ~Filters.command, process_start_message), CommandHandler('start', start)],
        states={
            Waiting_for_chat_message: [
                MessageHandler(Filters.text & ~Filters.command, process_message),
                MessageHandler(Filters.voice , process_voice_message),
                CommandHandler('help', help_command),CommandHandler('start', start)] + bot_commands,
            Waiting_for_raw_prompt: [
                MessageHandler(Filters.text & ~Filters.command,process_raw_prompt),
                CommandHandler('help', help_command)] + bot_commands,
        },
        fallbacks=[CommandHandler('help', help)],
        per_user=False,
    )

    dispatcher.add_handler(conv_handler)

    PORT = int(os.environ.get('PORT', '8443'))
    logger.debug("Using port " + str(PORT))

    # Start the Bot
    if 'WEBHOOK_URL' in os.environ:
        updater.start_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path=telegram_token,
            webhook_url=webhook_url + telegram_token
        )
        logger.info("Started webhook")
    else:
        updater.start_polling()
        logger.info("Started polling")
        
    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()

if __name__ == '__main__':
    main()
