from entity_extraction import get_entities
from text_preprocessing import get_corpus

# submodule
GENERAL_INTENT = 1
FAQS = 2
DEFAULT_REPLY = 3

if __name__ == '__main__':
    # Start of Chatbot
    print("Hi, how can I help you?")
    while True:
        user_message = input()  
        if user_message == "q":
            break 

        # Text Preprocessing
        processed_text=get_corpus(user_message)

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
            print(highest_confid_lvl_ans)

        elif match_type == FAQS:
            print(highest_confid_lvl_ans)

        elif match_type == DEFAULT_REPLY:
            print(highest_confid_lvl_ans)

    # End of Chatbot
    print("Thank you for using our chatbot.")
