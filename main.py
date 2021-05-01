# import libraries
from termcolor import colored
import logging
import argparse

logger = logging.getLogger(__name__)

# import other files
from entity_extraction import get_entities
from text_preprocessing import get_corpus as get_preprocessed_text
from database_extraction import read_database
from word_embedding import get_word_embedding
from scipy.spatial.distance import cosine

# submodule
GENERAL_INTENT = 1
FAQS = 2
DEFAULT_REPLY = 3

def preprocessed_whole_database(database, get_preprocessed_text):
    """
    To return a preprocessed version of questions in the database
    """
    orig2preprocessed_database = {}
    for submodule in database:
        for question, _ in submodule.items():
            orig2preprocessed_database[question] = get_preprocessed_text(question)

    return orig2preprocessed_database

def display_chatbot_reply(answer):
    """
    Print the chatbot reply message in color 
    """
    reply = "Chatbot: "
    reply += colored(answer,"green")
    print(reply)

def similarity_matching(preprocessed_user_message, candidates_submodules, get_word_embedding_func, default_reply, orig2preprocessed_database, default_reply_thres):
    """Match the user message and the candidate questions in database after LDA clustering
       to find the most similar question and answer along with the type of submodule 
    """
    user_message_embedding = get_word_embedding_func(preprocessed_user_message)
    max_similarity = 0
    max_submodule = 0
    max_question = ""
    max_answer = ""
    for submodule, candidate_questions in enumerate(candidates_submodules):
        for question, answer in candidate_questions.items():
            # the cosine formula in scipy is [1 - (u.v / (||u||*||v||))]
            # so we have to add 1 - consine() to become the similary match instead of difference match 
            similarity = 1 - cosine(user_message_embedding, get_word_embedding_func(orig2preprocessed_database[question]))
            if similarity > max_similarity:
                max_similarity, max_question, max_answer, max_submodule = similarity, question, answer, submodule + 1


    logger.info("Highest Matched Submodule: "+str(max_submodule))
    logger.info("Highest Similarity Score: "+str(max_similarity))
    logger.info("Highest Confidence Level Question: "+str(max_question))
    logger.info("Highest Confidence Level Answer: "+str(max_answer))

    # if the highest similarity is lower the predefined threshold
    # default reply will be sent back to the user
    if max_similarity >= default_reply_thres:
        return max_submodule, max_answer["answer"]
    else:
        return DEFAULT_REPLY, default_reply["answer"]

# database that contains Q&A
database = read_database()
orig2preprocessed_database = preprocessed_whole_database(database, get_preprocessed_text)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logging', action='store_true')
    parser.add_argument('--default_reply_thres', type=float, required=True)
    args = parser.parse_args()
    if args.logging:
        logging.basicConfig(level=logging.INFO)
        logger.setLevel(logging.INFO)

    # Start of Chatbot
    display_chatbot_reply("Hi, how can I help you?")
    while True:
        user_message = input()  
        if user_message == "q":
            break 

        # Text Preprocessing
        preprocessed_user_message = get_preprocessed_text(user_message)
        logger.info("Preprocessed User Message: "+str(preprocessed_user_message))

        # LDA + Clustering
        # TODO

        # Similarity Matching 
        # can replace database to candidate_database after LDA + Clustering(TODO)
        matched_submodule, highest_confid_lvl_ans = similarity_matching(preprocessed_user_message, database[:2], get_word_embedding, database[2]["Default"], orig2preprocessed_database, default_reply_thres=args.default_reply_thres)

        if matched_submodule == GENERAL_INTENT:
            # Entity Extraction 
            entities = get_entities(user_message)
            logger.info("Entities Extracted: "+str(entities))
            highest_confid_lvl_ans = highest_confid_lvl_ans.replace("PERSON", entities["PERSON"][0])
            highest_confid_lvl_ans = highest_confid_lvl_ans.replace("BANK_ACC", entities["BANK_ACC"][0])
            highest_confid_lvl_ans = highest_confid_lvl_ans.replace("AMOUNT", entities["AMOUNT"][0])
            display_chatbot_reply(highest_confid_lvl_ans)

        elif matched_submodule == FAQS:
            display_chatbot_reply(highest_confid_lvl_ans)

        elif matched_submodule == DEFAULT_REPLY:
            display_chatbot_reply(highest_confid_lvl_ans)

    # End of Chatbot
    display_chatbot_reply("Thank you for using our chatbot.")
