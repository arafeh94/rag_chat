from chat import ChatService

chat_service = ChatService.create()
chat_service.dummy() if False else ''
chat = chat_service.new_chat(v=2)
source = './files/2.pdf'.split(',')

chat2 = chat_service.new_chat(v=2)

while True:
    question = input("Question: ")
    if question == 'quit':
        break
    chat.answer(question).pretty_print()
    chat2.answer(question).pretty_print()
