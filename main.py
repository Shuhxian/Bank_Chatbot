# import libraries
from termcolor import colored

# import other files
from entity_extraction import get_entities
from text_preprocessing import get_corpus as get_preprocessed_text
from database_extraction import read_database
from lda import train_classifier, predict
from word_embedding import get_word_embedding
from scipy.spatial.distance import cosine

# submodule
GENERAL_INTENT = 1
FAQS = 2
DEFAULT_REPLY = 3

# database that contains Q&A
database = read_database()

def display_chatbot_reply(answer):
    """
    Print the chatbot reply message in color 
    """
    reply = "Chatbot: "
    reply += colored(answer,"green")
    print(reply)

def similarity_matching(preprocessed_user_message, candidates_submodules, get_word_embedding_func, default_reply, default_reply_thres = 0.5):
    """Match the user message and the candidate questions in database after LDA clustering
       to find the most similar question and answer along with the type of submodule 
    """
    user_message_embedding = get_word_embedding_func(preprocessed_user_message)
    max_similarity = 0
    max_submodule = 0
    max_answer = ""
    for submodule, candidate_questions in enumerate(candidates_submodules):
        for question, answer in candidate_questions:
            # the cosine formula in scipy is [1 - (u.v / (||u||*||v||))]
            # so we have to add 1 - consine() to become the similary match instead of difference match 
            similarity = 1 - cosine(user_message_embedding, get_word_embedding_func(question))
            if similarity > max_similarity:
                max_similarity, max_answer, max_submodule = similarity, answer, submodule + 1

    # if the highest similarity is lower the predefined threshold
    # default reply will be sent back to the user
    if max_similarity >= default_reply_thres:
        return max_submodule, max_answer
    else:
        return DEFAULT_REPLY, default_reply


if __name__ == '__main__':
    # Start of Chatbot
    display_chatbot_reply("Hi, how can I help you?")
    while True:
        user_message = input()  
        if user_message == "q":
            break 

        # Text Preprocessing
        preprocessed_user_message = get_preprocessed_text(user_message)

        # LDA + Clustering
        # TODO

        # Similarity Matching 
        # can replace database to candidate_database after LDA + Clustering
        matched_submodule, highest_confid_lvl_ans = similarity_matching(preprocessed_user_message, database[:2], get_word_embedding, database[2]["Default"])

        if matched_submodule == GENERAL_INTENT:
            # Entity Extraction 
            entities = get_entities(user_message)
            highest_confid_lvl_ans = highest_confid_lvl_ans.replace("NAME", entities["PERSON"][0])
            highest_confid_lvl_ans = highest_confid_lvl_ans.replace("BANK_ACC", entities["BANK_ACC"][0])
            highest_confid_lvl_ans = highest_confid_lvl_ans.replace("AMOUNT", entities["AMOUNT"][0])
            display_chatbot_reply(highest_confid_lvl_ans)

        elif matched_submodule == FAQS:
            display_chatbot_reply(highest_confid_lvl_ans)

        elif matched_submodule == DEFAULT_REPLY:
            display_chatbot_reply(highest_confid_lvl_ans)

    # End of Chatbot
    display_chatbot_reply("Thank you for using our chatbot.")
