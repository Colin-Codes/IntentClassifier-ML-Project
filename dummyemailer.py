import imapclient 
import pyzmail
import smtplib
import email

pa55word = input('Password: ')

iClient = imapclient.IMAPClient('imap.gmail.com')
iClient.login('hal9000.origin@gmail.com', pa55word)
iClient.select_folder('INBOX')
messageIDs = iClient.search(['ALL'])
messageBodies = iClient.fetch(messageIDs, ['BODY[]'])

for messageID in messageIDs:
    messageBody = pyzmail.PyzMessage.factory(messageBodies[messageID][b'BODY[]'])
    print('Subject: ' + messageBody.get_subject())
    print('From: ' + str(messageBody.get_addresses('from')))
    print('To: ' + str(messageBody.get_addresses('to')))
    print('CC: ' + str(messageBody.get_addresses('cc')))
    if messageBody.text_part != None:
        print('Body: ' + messageBody.text_part.get_payload().decode(messageBody.text_part.charset))
    else:
        print('Body: ' + messageBody.text_part.get_payload().decode(messageBody.html_part.charset))
    for emailAddress in messageBody.get_addresses('from'):
        if emailAddress[1] != 'colin.younge@gmail.com' and emailAddress[1] != 'colin.younge@googlemail.com':
            continue
        s = smtplib.SMTP('smtp.gmail.com', 587)
        s.starttls()
        s.ehlo()
        s.login('hal9000.origin@gmail.com', pa55word)
        text = """Hello, World! 
        I can't let you do that Dave... """
        msg = email.message.EmailMessage()
        msg['from'] = 'hal9000.origin@gmail.com'
        msg["to"] = emailAddress[1]
        msg["Subject"] = "Re: Hello World! "
        msg.set_content(text)
        res = s.send_message(msg)