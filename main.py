from termcolor import colored
from entity_extraction import get_entities
from text_preprocessing import get_corpus as get_preprocessed_text

# submodule
GENERAL_INTENT = 1
FAQS = 2
DEFAULT_REPLY = 3

def display_chatbot_reply(reply):
    reply = "Chatbot: "
    reply += colored(reply+"\n","green")
    print(reply)

if __name__ == '__main__':
    # Start of Chatbot
    display_chatbot_reply("Hi, how can I help you? \n")
    while True:
        user_message = input()  
        if user_message == "q":
            break 

        # Text Preprocessing
        processed_text = get_preprocessed_text(user_message)

        # Word Embedding
        # TODO

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
