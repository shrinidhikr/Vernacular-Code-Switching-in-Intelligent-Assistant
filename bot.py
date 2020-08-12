#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This program is dedicated to the public domain under the CC0 license.

"""
Simple Bot to reply to Telegram messages.
First, a few handler functions are defined. Then, those functions are passed to
the Dispatcher and registered at their respective places.
Then, the bot is started and runs until we press Ctrl-C on the command line.
Usage:
Basic Echobot example, repeats messages.
Press Ctrl-C on the command line or send a signal to the process to stop the
bot.
"""

import logging
import pipeline_covid
import imp
imp.reload(pipeline_covid)
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from telegram import Location, KeyboardButton, ReplyKeyboardMarkup
# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


# Define a few command handlers. These usually take the two arguments update and
# context. Error handlers also receive the raised TelegramError object in error.
def start(update, context):
    """Send a message when the command /start is issued."""
    update.message.reply_text('Hi! How can I help you')


def help(update, context):
    """Send a message when the command /help is issued."""
    location_keyboard = KeyboardButton(text="Send location",  request_location=True)
    contact_keyboard = KeyboardButton('Send contact info', request_contact=True) 
    custom_keyboard = [[ location_keyboard, contact_keyboard ]] 
    reply_markup = ReplyKeyboardMarkup(custom_keyboard) 
    update.message.reply_text("Would you mind sharing your location and contact with me? So that I can help you!",reply_markup=reply_markup)
    #print(update)

def echo(update, context):
    """Echo the user message."""
    #print("Update:   ",update)

    update.message.reply_text(pipeline_covid.perform_action(update.message.text))


def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)


def main():
    """Start the bot."""
    print("Starting the bot")
    # Create the Updater and pass it your bot's token.
    # Make sure to set use_context=True to use the new context based callbacks
    # Post version 12 this will no longer be necessary
    # 1296565123:AAEmIF0FulQ0-YWHP0T8JpG56rK23mg4Sf4 - trialbot

    # updater = Updater("1107228750:AAGjyQqaNRFJbL4HjBoar5WklS9zXRvlj-s", use_context=True)
    updater = Updater("1217886685:AAHD8Dr74VgQBfw6H7-yEd1Y_pGOkKmw26U", use_context=True)
    
    #print("Updater:   ",updater)
    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # on different commands - answer in Telegram
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))

    # on noncommand i.e message - echo the message on Telegram
    dp.add_handler(MessageHandler(Filters.text, echo))

    # log all errors
    dp.add_error_handler(error)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()
