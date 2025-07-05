from twilio.rest import Client

# Your Twilio credentials
account_sid = 'A#######ef8f2bcd66b6ac7c6f3bafced2af'
auth_token = '@###@#@@@@3b58e48c41329198caba'

client = Client(account_sid, auth_token)

message = client.messages.create(
    body='hey your frd is in risk , call him up!!!!.... ',
    from_='+00000017983',  # Your Twilio number
    to='+9==554443064'      # Recipient's number
)

print(f"Message sent with SID: {message.sid}")
