from flask import Flask, request, jsonify
from keybert import KeyBERT
import numpy as np
from random import randint

"""
flask=2.2.2
keybert==0.7.0
maybe numpy. It should be installed with keybert


Below is a list of the possible pages the Quiz Feedback will tell you to visit:
       'Real life examples'
       'The definition of fake news'
       'The existence of fake news/Biased Information'
       'The importance of identifying fake news'
       'Tips to spot fake news'
       'Tips to spot fake news/How can you spot biased news'
"""

app = Flask(__name__)


@app.route('/get_keywords', methods=["POST"])
def keywords_GET():
    text = request.json['text']
    try:
        max_keywords = request.json['max_keywords']
    except:
        max_keywords = None
    # Processong on text
    keywords = generate_keywords(text, max_keywords)

    return jsonify(keywords)


@app.route('/get_keywords/<string:text>', methods=["GET"])
def keywords_POST(text):
    keywords = generate_keywords(text)

    return jsonify(keywords)


def generate_keywords(text, max_keywords=None):
    """
    Arguments:
            docs: The document(s) for which to extract keywords/keyphrases
            candidates: Candidate keywords/keyphrases to use instead of extracting them from the document(s)
                        NOTE: This is not used if you passed a `vectorizer`.
            keyphrase_ngram_range: Length, in words, of the extracted keywords/keyphrases.
                                   NOTE: This is not used if you passed a `vectorizer`.
            stop_words: Stopwords to remove from the document.
                        NOTE: This is not used if you passed a `vectorizer`.
            top_n: Return the top n keywords/keyphrases
            min_df: Minimum document frequency of a word across all documents
                    if keywords for multiple documents need to be extracted.
                    NOTE: This is not used if you passed a `vectorizer`.
            use_maxsum: Whether to use Max Sum Distance for the selection
                        of keywords/keyphrases.
            use_mmr: Whether to use Maximal Marginal Relevance (MMR) for the
                     selection of keywords/keyphrases.
            diversity: The diversity of the results between 0 and 1 if `use_mmr`
                       is set to True.
            nr_candidates: The number of candidates to consider if `use_maxsum` is
                           set to True.
            vectorizer: Pass in your own `CountVectorizer` from
                        `sklearn.feature_extraction.text.CountVectorizer`
            highlight: Whether to print the document and highlight its keywords/keyphrases.
                       NOTE: This does not work if multiple documents are passed.
            seed_keywords: Seed keywords that may guide the extraction of keywords by
                           steering the similarities towards the seeded keywords.
            doc_embeddings: The embeddings of each document.
            word_embeddings: The embeddings of each potential keyword/keyphrase across
                             across the vocabulary of the set of input documents.
                             NOTE: The `word_embeddings` should be generated through
                             `.extract_embeddings` as the order of these embeddings depend
                             on the vectorizer that was used to generate its vocabulary.
    """
    if max_keywords:
        return kw_model.extract_keywords(docs=[text], keyphrase_ngram_range=(1, max_keywords), top_n=30)
    return kw_model.extract_keywords(docs=[text], keyphrase_ngram_range=(1, len(text)), top_n=30)


@app.route('/quiz_feedback', methods=["POST"])
def get_quiz_feedback():
    try:
        q = []
        a = []

        N_QUESTIONS = 8
        for i in range(1, N_QUESTIONS + 1):
            q.append(request.json[f"q{i}"])
            a.append(request.json[f"a{i}"])

    except Exception as e:
        return jsonify({"Error": str(e),
                        "Message": "There is an issue with the payload/input JSON file from your end.  The json file should have 16 parameters.  The first eight are 'q1' to 'q8', then there's 'a1' to 'a8'.  The q's represent the QID (from the Quiz Google Doc), where the a's represent the answer the user gave as an integer (indexed from zero)."})

    try:
        assert len(q) == len(
            set(q)), "The quiz feedback does not work if we have duplicate questions.  Please check that all questions are unique, and we did not ask the same question twice."
    except Exception as e:
        return jsonify({"Error": str(e),
                        "Message": "Please check error message."})

    try:
        feedback, page_to_visit = get_quiz_feedback(q, a)
    except Exception as e:
        return jsonify({"Error": str(e),
                        "Message": "An error has occurred within the Python code."})

    return jsonify({"Feedback": feedback, "Suggested Page": page_to_visit})


class Question():
    def __init__(self, question, question_type, question_id=None):
        self.question = question
        self.answers = []
        self.correct_answer = None
        self.type = question_type
        self.learning_outcomes = dict()

        if question_id:
            self.question_id = question_id
        else:
            self.question_id = hash(question)

    def add_answer(self, answer, correct, learning_outcome: list):
        if self.correct_answer and correct:
            raise AssertionError("Only one answer can be correct.")

        if correct:
            self.correct_answer = len(self.answers)
        self.answers.append(answer)

        assert len(learning_outcome) == 6
        self.learning_outcomes[answer] = learning_outcome

    def get_learning_outcome_score(self, answer):
        if type(answer) is int:
            # Get answer
            answer = self.answers[answer]

        learning_outcome = self.learning_outcomes[answer]

        return learning_outcome

    def get_question_id(self):
        return self.question_id

    def get_question(self):
        return self.question

    def get_number_of_answers(self):
        return len(self.answers)


class QuestionBank():
    def __init__(self):
        self.questions = []
        self.id_to_q = dict()

    def add_question(self, question: Question):
        self.questions.append(question)
        self.id_to_q[question.get_question_id()] = question

    def get_question(self, question_number: int):
        return self.questions[question_number]

    def id_to_question(self, id_):
        return self.id_to_q.get(id_)

    def get_all_questions(self):
        return self.questions


class Quiz():
    def __init__(self, question_bank):
        self.question_bank = question_bank
        self.quiz_questions = []
        self.quiz_answers = []

    def add_questions(self, questions):
        for q in questions:
            if type(q) is str:
                question = self.question_bank.id_to_question(q)
                self.quiz_questions.append(question)

            elif type(q) is Question:
                self.quiz_questions.append(q)

            else:
                raise AssertionError(f"Invalid Question: {q}")

            self.quiz_answers.append(None)

    def add_question(self, question):
        if type(question) is str:
            q = self.question_bank.id_to_question(question)
            if q is None:
                raise AssertionError(f"Question ID is not valid: {question}")
            self.quiz_questions.append(q)

        elif type(question) is Question:
            self.quiz_questions.append(question)

        else:
            raise AssertionError("Invalid Question")

        self.quiz_answers.append(None)

    def add_answer(self, question, answer):
        if type(question) is str:
            q = self.question_bank.id_to_question(question)
            if q is None:
                raise AssertionError(f"Question ID is not valid: {question}")
            question = q

        if type(question) != int:
            question_index = self.quiz_questions.index(question)
        else:
            question_index = question

        self.quiz_answers[question_index] = answer

    def score_quiz(self):
        score = np.asarray([0] * len(learning_outcomes))
        for q, a in zip(self.quiz_questions, self.quiz_answers):
            if a is None:
                raise AssertionError("Not all quiz questions have answers")
            try:
                question_score = q.get_learning_outcome_score(a)
            except IndexError as e:
                raise IndexError(f"""For question {q.get_question_id()}, the answer {a} is not valid.
                                    All quiz question's and answers are stored in the Python file.  If the answers were not typed properly into the python file, or the answers are incomplete, this error could occur.  Otherwise, check that the answer you sent is valid.  Answers should be an integer, where the index starts at 0.""")
            question_score = np.asarray(question_score)
            score += question_score

        return score

    def get_questions(self):
        return self.quiz_questions


learning_outcomes = [
    "Understanding 'What is fake news'",
    "Understanding 'what is biased news'",
    "Understanding how to identify 'biased information'",
    "Recognising intentional fake news",
    "Recognising unintentional fake news",
    "Understanding what are some consequences of falling for Fake News",
]


def create_question_bank():
    bank = QuestionBank()

    # Question 1
    question = Question("Which of the following is an effective way to spot fake news", question_type="MCS",
                        question_id="MC1")
    question.add_answer(answer="Believing every article you read",
                        correct=False,
                        learning_outcome=[-3, -1, -1, -5, -5, 0]
                        )
    question.add_answer(answer="Checking multiple sources for accuracy",
                        correct=True,
                        learning_outcome=[2, 0, 1, 3, 3, 2]
                        )
    question.add_answer(answer="Sharing articles without reading them",
                        correct=False,
                        learning_outcome=[-1, -1, -1, -2, -2, -5]
                        )
    bank.add_question(question)

    # Question 2
    question = Question("Which of the following is a reliable source for news?", question_type="MCS", question_id="MC2")
    question.add_answer(answer="Your friend's blog",
                        correct=False,
                        learning_outcome=[-1, -1, -1, -2, -1, -1]
                        )
    question.add_answer(answer="A well established newspaper",
                        correct=True,
                        learning_outcome=[0, 0, 0, 0, 2, 0],
                        )
    question.add_answer(answer="A random website with no credentials or background information",
                        correct=False,
                        learning_outcome=[-2, -3, -1, -2, 2, -1]
                        )
    bank.add_question(question)

    # Question 3
    question = Question("Which of the following is an example of a fact-checking website?", question_type="MCS",
                        question_id="MC3")
    question.add_answer(answer="Twitter",
                        correct=False,
                        learning_outcome=[-5, -3, -5, -4, -4, -2]
                        )
    question.add_answer(answer="Snopes",
                        correct=True,
                        learning_outcome=[5, 4, 5, 3, 3, 0],
                        )
    question.add_answer(answer="Reddit",
                        correct=False,
                        learning_outcome=[-5, -3, -5, -4, -4, -2]
                        )
    bank.add_question(question)

    # Question 4
    question = Question("Which of the following should you do if you suspect an article is fake news?",
                        question_type="MCS", question_id="MC4")
    question.add_answer(answer="Share the article on social media to warn others",
                        correct=False,
                        learning_outcome=[-1, -1, -2, -1, -1, -5]
                        )
    question.add_answer(answer="Ignore the article and move on",
                        correct=False,
                        learning_outcome=[0, 0, 0, 0, 0, 0],
                        )
    question.add_answer(answer="Research the topic and look for credible sources to verify information",
                        correct=True,
                        learning_outcome=[0, 0, 0, 0, 0, -5]
                        )
    bank.add_question(question)

    # Question 5
    question = Question("Which of the following is an example of a reliable source for politcal news?",
                        question_type="MCS", question_id="MC5")
    question.add_answer(answer="A partesian political blog that only covers one side of the political spectrum",
                        correct=False,
                        learning_outcome=[0, -4, -5, 0, 0, -2],
                        )
    question.add_answer(answer="A well-respected news outlet that has a reputation for unbias reporting",
                        correct=True,
                        learning_outcome=[0, 4, 5, 0, 0, 0]
                        )
    question.add_answer(answer="A random website with no credentials or background information",
                        correct=False,
                        learning_outcome=[0, -2, -3, 0, 0, -2]
                        )
    bank.add_question(question)

    # Question 6
    question = Question("Which of the following is a common red flag that may indicate a news article is fake?",
                        question_type="MCS", question_id="MC6")
    question.add_answer(answer="The article uses proper grammar and punctuation.",
                        correct=False,
                        learning_outcome=[0, 0, -2, -5, -5, 0],
                        )
    question.add_answer(answer="The article contains quotes from credible sources",
                        correct=False,
                        learning_outcome=[-3, -1, 0, -4, -4, -1]
                        )
    question.add_answer(answer="The article includes a sensational or clickbait headline",
                        correct=True,
                        learning_outcome=[2, 2, 3, 3, 3, 0]
                        )
    bank.add_question(question)

    # Question 7
    question = Question("Which of the following headlines is most likely to be fake news?", question_type="MCS",
                        question_id="MC7")
    question.add_answer(answer="Study shows that coffee can reduce the risk of cancer.",
                        correct=False,
                        learning_outcome=[-5, -2, 0, -4, 0, -1]
                        )
    question.add_answer(answer="Alien spaceship found on Mars",
                        correct=True,
                        learning_outcome=[3, 2, 1, 5, 4, 0]
                        )
    question.add_answer(answer="""New survey reveals that 70% of people prefer ice cream over cake.""",
                        correct=False,
                        learning_outcome=[4, 5, 5, 2, 1, 0]
                        )
    bank.add_question(question)

    # Question 8
    question = Question(
        "Which of the following is a common technique used in fake news articles to manipulate readers?",
        question_type="MCS", question_id="MC8")
    question.add_answer(answer="Using credible sources and providing factual information.",
                        correct=False,
                        learning_outcome=[-5, -4, 0, -2, -1, 0]
                        )
    question.add_answer(answer="Using emotional language and appealing to readers' emotions.",
                        correct=True,
                        learning_outcome=[3, 2, 5, 5, 2, 0]
                        )
    question.add_answer(answer="Using objective language and presenting multiple viewpoints.",
                        correct=False,
                        learning_outcome=[0, 5, 5, 3, 3, 4]
                        )
    bank.add_question(question)

    """
    PART 2: WHICH ONE IS FAKE
    """
    # Question 1:
    question = Question("Which article below could be fake?", question_type="Fake Article", question_id="A1")
    question.add_answer(answer="Article 1",
                        correct=False,
                        learning_outcome=[-5, 0, 0, -2, -2, 0]
                        )
    question.add_answer(answer="Article 2",
                        correct=True,
                        learning_outcome=[2, 0, 0, 3, 0, 0],
                        )
    bank.add_question(question)

    # Question 2:
    question = Question("Which article below could be fake?", question_type="Fake Article", question_id="A2")
    question.add_answer(answer="Article 1",
                        correct=True,
                        learning_outcome=[3, 0, 0, 3, 1, 1]
                        )
    question.add_answer(answer="Article 2",
                        correct=False,
                        learning_outcome=[-3, 0, 0, -4, 1, -1],
                        )
    bank.add_question(question)
    # Question 3:
    question = Question("Which article below could be fake?", question_type="Fake Article", question_id="A3")
    question.add_answer(answer="Article 1",
                        correct=True,
                        learning_outcome = [2, 0, 0, 4, 1, 0]
                        )
    question.add_answer(answer="Article 2",
                        correct=False,
                        learning_outcome = [-1, 0, 0, -2, -1, -1],
                        )
    bank.add_question(question)

    return bank


class Feedback():
    def __init__(self, results):
        self.results = results
        self.learning_outcomes = ["Understanding 'What is fake news'",
                                  "Understanding 'what is biased news'",
                                  "Understanding how to identify 'biased information'",
                                  'Recognising intentional fake news',
                                  'Recognising unintentional fake news',
                                  'Understanding what are some consequences of falling for Fake News']
        # Ranks learning outcomes from best to worst
        self.learning_outcomes_ranks = np.argsort(results)[::-1]

    def construct(self):
        output = ""
        output += self.intro_to_positive() + " "
        output += self.first_positive()
        output += " and "
        output += self.second_positive() + ".  "

        improvement_intro = self.area_of_improvement_intro()
        if type(improvement_intro) is tuple:
            output += improvement_intro[0] + " "
            output += self.biggest_weakness() + " "
            output += improvement_intro[1]
        else:
            output += improvement_intro + " "
            output += self.biggest_weakness()
        output += ". "
        output += self.improvement_advice() + "\n"

        page, text = self.recommended_reading()
        output += text

        return output, page

    def intro_to_positive(self):
        intro_to_good = [
            "It's great to see that you've got a good grasp on",
            "From your quiz results, we can see that you're great at",
            "It seems like you're already good at",
        ]
        i = randint(0, len(intro_to_good) - 1)
        return intro_to_good[i]

    def first_positive(self):
        feedback_title = {
            0: ["understanding what fake news is", "knowing what fake news is"],
            1: ["understanding what biased news is", "understanding what biased information looks like"],
            2: ["knowing how to identify biased information online", "being able to identify biased information"],
            3: ["being able to recognise intentional fake news", "recognising intentional fake news",
                "being able to recognise fake news which had been created intentionally"],
            4: ["being able to recognise unintentional fake news", 'recognising unintentional fake news'],
            5: ["understanding some of the consequences of falling for fake news",
                "understanding some of the issues associated with believing fake news",
                "knowing the consequences of falling for fake news"],
        }
        possible_statements = feedback_title[self.learning_outcomes_ranks[0]]

        i = randint(0, len(possible_statements) - 1)
        return possible_statements[i]

    def second_positive(self):
        feedback_title = {
            0: ["understanding what fake news is", "knowing what fake news is"],
            1: ["understanding what biased news is", "understanding what biased information looks like"],
            2: ["knowing how to identify biased information online", "being able to identify biased information"],
            3: ["being able to recognise intentional fake news", "recognising intentional fake news",
                "being able to recognise fake news which had been created intentionally"],
            4: ["being able to recognise unintentional fake news", 'recognising unintentional fake news'],
            5: ["understanding some of the consequences of falling for fake news",
                "understanding some of the issues associated with believing fake news",
                "knowing the consequences of falling for fake news"],
        }
        possible_statements = feedback_title[self.learning_outcomes_ranks[1]]

        i = randint(0, len(possible_statements) - 1)
        return possible_statements[i]

    def biggest_weakness(self):
        feedback_title = {
            0: ["understanding what fake news is", "knowing what fake news is"],
            1: ["understanding what biased news is", "understanding what biased information looks like"],
            2: ["knowing how to identify biased information online", "being able to identify biased information"],
            3: ["being able to recognise intentional fake news", "recognising intentional fake news",
                "being able to recognise fake news which had been created intentionally"],
            4: ["being able to recognise unintentional fake news", 'recognising unintentional fake news'],
            5: ["understanding some of the consequences of falling for fake news",
                "understanding some of the issues associated with believing fake news",
                "knowing the consequences of falling for fake news"],
        }
        possible_statements = feedback_title[self.learning_outcomes_ranks[-1]]
        i = randint(0, len(possible_statements) - 1)
        return possible_statements[i]

    def area_of_improvement_intro(self):
        improvement = [
            ("However, it seems like", "is a challenge for you"),
            ("However, it seems like you would benefit from working on"),
            ("Although there's room for improvement on"),
            ("But unfortunately you scored lowly on"),
            ("However, it could be worth you spending some time on",
             "so that you're better equipped to avoid believing misinformation"),
        ]
        i = randint(0, len(improvement) - 1)
        return improvement[i]

    def improvement_advice(self):
        how_to_improve = {
            # Understanding "what is fake news"
            0: ["You can improve this by researching different types of fake news.",
                "You can improve on this by learning about different sources.  Having a strong understanding on the key differences between quality reputable sources, and sources with no accountability."],

            # Understanding 'what is biased news'
            1: [
                "You should look into some example's of far left media websites, and far right media websites.  Try having a read through some of their articles and see if you can identify any key differences.",
            ],

            # Understanding how to identify biased information
            2: [
                "You should try learning about 'story selection bias,' which can help you understand how various media sources try and shape your worldview by the type of stories they publish.",
                "You should focus on trying to understand how to identify different loaded words and the emotions the author tries to envoke when they use them.",
            ],

            # Recognising intentional fake news
            3: [
                "You can improve this by learning about different types of sources and knowing how to identify quality reputable organisations versus sources that have no accountability.",
                "You can improve this by learning to identify the agenda behind different online posts or news articles.",

            ],

            # Recognising unintentional fake news
            4: [
                "You can improve this by learning about different political biases people have, and how they influence the type of information individual's choose to share online.",

            ],

            # Understanding what are some consequences of falling for fake news
            5: [
                "It won't always be an issue if we believe some misinformation online, however there is always a danger associated with it.",
                "Educating yourself on what some of the consequences could be will help you learn the imporance of recognising misinformation."
            ],
        }

        possible_statements = how_to_improve[self.learning_outcomes_ranks[-1]]
        i = randint(0, len(possible_statements) - 1)
        return possible_statements[i]

    def recommended_reading(self):
        page_to_read = {
            0: [("The definition of fake news",
                 "Based on your results, we recommend you read the following page to learn the different types of fake news."),
                ("Real life examples",
                 "From your results, we recommend you read though some of our real-life examples on fake news.  By seeing some example's on fake news, you can develop a better understanding on what it is and what it looks like."),
                ("The definition of fake news",
                 "We recommend you check out our page on the definition of fake news to gain a better understanding on the differences between various kinds of fake news."),
                ("The definition of fake news",
                 "To further develop your understanding, have a look at the following educational page to help gain a better overview on what fake news is.")
                ],

            1: [("The existence of fake news/Biased Information",
                 "Based on these results, we recommend you have a look at the following page.  This page should help you gain an understanding on how news stories can contain bias, even if the author's were not intentional about adding it."),
                ("The existence of fake news/Biased Information",
                 "We recommend you have a look at our page on why fake news is created.   This should help clarify what biased news is and how it gets created."),
                ],

            2: [("Tips to spot fake news/How can you spot biased news",
                 "Based on your results, we recommend you read the following page to learn about how you can recognise bias in the media."),
                ("Tips to spot fake news/How can you spot biased news",
                 "Have a look at the following page.  This should by learning some extra tips on recognising media bias, you'll be able to help prevent yourself from falling for this form of misinformation.")
                ],

            3: [("Tips to spot fake news",
                 "As it seems like you havce trouble recognising fake news, we recommend you read our page on spotting fake news."),
                ("Tips to spot fake news",
                 "From your quiz results, we recommend you have a look at our article on tips to spot fake news.  This should help you become more comfortable in recognising intentiona fake news."),
                ("Real life examples",
                 "As it seems you're having trouble recognising unintentional misinformation online, we recommend you have a look at our real life example's page.  Here, you can see some fake news stories which have spread online, and we give you tips on how this story could have been identified as being fake news."),

                ],

            4: [("Tips to spot fake news",
                 "Have a look at our article on spotting fake news.  This should help give you some tips and ideas on the best ways to spot fake news and prevent yourself from falling for misinformation."),
                ("Real life examples",
                 "As it seems you're having trouble recognising intentional misinformation online, we recommend you have a look at our real life example's page.  Here, you can see some fake news stories which have spread online, and we give you tips on how this story could have been identified as being fake news."),
                ],

            5: [("The importance of identifying fake news",
                 "Given your low score in understanding the consequences of believing misinformation online, we suggest that you read the following page which outline's the key reasons why it's important to ensure you're able to identify fake news and help prevent it spreading."),
                ("The importance of identifying fake news",
                 "To further develop your understanding, have a look at the following article on the importance of identifying fake news.  This article explain's why fake news can be a serious issue when it spreads, and it will help you gain an understanding on why it will benefit you to be able to recognise fake news.")],
        }
        possible_statements = page_to_read[self.learning_outcomes_ranks[-1]]
        i = randint(0, len(possible_statements) - 1)
        return possible_statements[i]


def get_quiz_feedback(questions: list, answers: list):
    bank = create_question_bank()
    quiz = Quiz(bank)
    for question, answer in zip(questions, answers):
        quiz.add_question(question)
        quiz.add_answer(question, answer)

    scores = quiz.score_quiz()
    fb = Feedback(scores)
    feedback, page_to_visit = fb.construct()
    return feedback, page_to_visit

if __name__ == '__main__':
    kw_model = KeyBERT()
    app.run('0.0.0.0',port=5000)
