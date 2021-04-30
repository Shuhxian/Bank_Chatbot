
# submodule
GENERAL_INTENT = 1
FAQS = 2
DEFAULT_REPLY = 3

if __name__ == '__main__':
    while True:
        user_message = input("Hi, how can I help you?")  
        if user_message == "q":
            break 

        # Text Preprocessing
        # TODO

        # Word Embedding
        # TODO

        # LDA + Similarity Matching 
        # TODO

        if match_type == GENERAL_INTENT:
            # Entity Extraction 
            # TODO
            highest_confid_lvl_ans = highest_confid_lvl_ans.replace("NAME", NAME)
            highest_confid_lvl_ans = highest_confid_lvl_ans.replace("BANK_ACC", BANK_ACC)
            highest_confid_lvl_ans = highest_confid_lvl_ans.replace("AMOUNT", AMOUNT)
            return highest_confid_lvl_ans

        elif match_type == FAQS:
            return highest_confid_lvl_ans

        elif match_type == DEFAULT_REPLY:
            return highest_confid_lvl_ans

    # End of Chatbot
    print("Thank you for using our chatbot.")
