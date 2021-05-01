# import libraries
from termcolor import colored

# import other files
from entity_extraction import get_entities
from text_preprocessing import get_corpus as get_preprocessed_text
from database_extraction import read_database
from lda import train_classifier, predict

# submodule
GENERAL_INTENT = 1
FAQS = 2
DEFAULT_REPLY = 3

# database that contains Q&A
database = read_database()

def display_chatbot_reply(answer):
    reply = "Chatbot: "
    reply += colored(answer,"green")
    print(reply)

if __name__ == '__main__':
    # Start of Chatbot
    display_chatbot_reply("Hi, how can I help you?")
    while True:
        user_message = input()  
        if user_message == "q":
            break 

        # Text Preprocessing
        processed_text = get_preprocessed_text(user_message)

        # LDA + Similarity Matching 
        # TODO

        if match_type == GENERAL_INTENT:
            # Entity Extraction 
            entities = get_entities(user_message)
            highest_confid_lvl_ans = highest_confid_lvl_ans.replace("NAME", entities["PERSON"][0])
            highest_confid_lvl_ans = highest_confid_lvl_ans.replace("BANK_ACC", entities["BANK_ACC"][0])
            highest_confid_lvl_ans = highest_confid_lvl_ans.replace("AMOUNT", entities["AMOUNT"][0])
            display_chatbot_reply(highest_confid_lvl_ans)

        elif match_type == FAQS:
            display_chatbot_reply(highest_confid_lvl_ans)

        elif match_type == DEFAULT_REPLY:
            display_chatbot_reply(highest_confid_lvl_ans)

    # End of Chatbot
    display_chatbot_reply("Thank you for using our chatbot.")
