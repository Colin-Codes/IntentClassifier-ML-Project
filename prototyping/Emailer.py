import imapclient 
import pyzmail
import smtplib
import email
import getpass
import Predictor

pa55word = getpass.getpass()

iClient = imapclient.IMAPClient('imap.gmail.com')
iClient.login('hal9000.origin@gmail.com', pa55word)
iClient.select_folder('INBOX')
messageIDs = iClient.search(['ALL'])
messageBodies = iClient.fetch(messageIDs, ['BODY[]'])

for messageID in messageIDs:
    messageBody = pyzmail.PyzMessage.factory(messageBodies[messageID][b'BODY[]'])
    subject = messageBody.get_subject()
    allowedAddresses = ['colin.younge@gmail.com', 'colin.younge@googlemail.com', 'colin.younge@origin-global.com']
    fromAddresses = [address[1] for address in messageBody.get_addresses('from') if address[1] in allowedAddresses ]
    toAddresses = messageBody.get_addresses('to')
    ccAddresses = messageBody.get_addresses('cc')
    if messageBody.text_part != None:
        body = messageBody.text_part.get_payload().decode(messageBody.text_part.charset)
    else:
        body = messageBody.text_part.get_payload().decode(messageBody.html_part.charset)
    print('Subject: ' + subject)
    print('From: ' + str(fromAddresses))
    print('To: ' + str(toAddresses))
    print('CC: ' + str(ccAddresses))
    print('Body: ' + body)

    for fromAddress in fromAddresses:
        s = smtplib.SMTP('smtp.gmail.com', 587)
        s.starttls()
        s.ehlo()
        s.login('hal9000.origin@gmail.com', pa55word)

        prediction = Predictor.predict(body)
        print(prediction)
        if prediction[0,0] > 0.5:
            text = "The weight of the window frames is x" 
        elif prediction[0,1] > 0.5:
            text = "The project has become invalidated and we will investigate" 
        elif prediction[0,2] > 0.5:
            text = "The average document generation time is x" 
        elif prediction[0,3] > 0.5:
            text = "You need to reset your password" 
        else:
            continue

        msg = email.message.EmailMessage()
        msg['from'] = 'hal9000.origin@gmail.com'
        msg["to"] = fromAddress
        msg["Subject"] = "Re: " + subject
        msg.set_content(text + "\n\n ------------------------------------------------------- \n\nYour original message: \n\n" + body)
        res = s.send_message(msg)