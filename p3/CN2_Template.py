# =============================================================================
# HOMEWORK 3 - RULE-BASED LEARNING
# CN2 ALGORITHM TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email me: arislaza@csd.auth.gr
# =============================================================================


# For this project, the only thing that we will need to import is the "Orange" library.
# However, before importing it, you must first install the library into Python.
# Read the instructions on how to do that (it might be a bit trickier than usual!)
# =============================================================================


# IMPORT LIBRARY HERE (trivial but necessary...)
import Orange

# =============================================================================



# Load 'wine' dataset
# =============================================================================


# ADD COMMAND TO LOAD TRAIN AND TEST DATA HERE
wineData = Orange.data.Table("wine")
# =============================================================================

#print(wineData.domain)


# Define the learner that will be trained with the data.
# Try two different learners: an '(Ordered) Learner' and an 'UnorderedLearner'.
# =============================================================================


# ADD COMMAND TO DEFINE LEARNER HERE
#learner = 
cn2o_learner = Orange.classification.CN2Learner()

cn2u_learner = Orange.classification.CN2UnorderedLearner()
cn2u_learner.rule_finder.search_strategy.constrain_continuous = True
cn2Laplace_learner = Orange.classification.CN2Learner()
# =============================================================================

c = Orange.evaluation.CrossValidation(stratified=True)


# At this step we shall configure the parameters of our learner.
# We can set the evaluator/heuristic ('Entropy', 'Laplace' or 'WRAcc'),
# 'beam_width' (in the range of 3-10), 'min_covered_examples' (start from 7-8 and make your way up),
# and 'max_rule_length' (usual values are in the range of 2-5).
# They are located deep inside the 'learner', within the 'rule_finder' class.
# Note: for the evaluator, set it using one of the Evaluator classes in classification.rules
# =============================================================================


# ADD COMMANDS TO CONFIGURE THE LEARNER HERE



# with below code i checked which combination provides better accurancy and i choose one of them as the optimal combination, due to highest accurancy, i needed at least 70min to run the below code, so i comment it

# =============================================================================
# classifier =[]
#  
# for i in range(3,12):
#     cn2u_learner.rule_finder.search_algorithm.beam_width = i
#     for j in range(3,12):
#         cn2u_learner.rule_finder.general_validator.min_covered_examples = i
#         for k in range(3,13):
#           cn2u_learner.rule_finder.general_validator.max_rule_length = i
#           classifier.append(c(wineData, [cn2u_learner]))
#       
# for i in range(len(classifier)):  
#     accuracy = Orange.evaluation.scoring.CA(classifier[i])
#     F1 = Orange.evaluation.scoring.F1(classifier[i])
#     for  acc, f1 in  zip(accuracy,F1):
#      print("Accuracy and F1"+acc+": " + str(acc) + " , " +str(f1))
#      print()
# =============================================================================
   
   

# =============================================================================

#after hours and testing with accuracy i chosed the below as the best combination of values

#ordered learner:

cn2o_learner.rule_finder.search_algorithm.beam_width = 6
cn2o_learner.rule_finder.general_validator.min_covered_examples = 15
cn2o_learner.rule_finder.general_validator.max_rule_length = 4

#unordered learner:

cn2u_learner.rule_finder.search_algorithm.beam_width = 2
cn2u_learner.rule_finder.general_validator.min_covered_examples = 5
cn2u_learner.rule_finder.general_validator.max_rule_length = 4

#ordered laplace:
cn2Laplace_learner.rule_finder.quality_evaluator = Orange.classification.rules.LaplaceAccuracyEvaluator()
cn2Laplace_learner.rule_finder.search_algorithm.beam_width = 12
cn2Laplace_learner.rule_finder.general_validator.min_covered_examples = 8
cn2Laplace_learner.rule_finder.general_validator.max_rule_length = 4
            

# =============================================================================




# We want to test our model now. The CrossValidation() function will do all the
# work in this case, which includes splitting the whole dataset into train and test subsets,
# then train the model, and produce results.
# So, simply initialize the CrossValidation() object from the 'testing' library
# and call it with input arguments 1) the dataset and 2) the learner.
# Note that the 'learner' argument should be in array form, i.e. '[learner]'.
cv = Orange.evaluation.CrossValidation(stratified=True)


#list of lerners

cn2o_results = cv(wineData,[cn2o_learner])
cn2u_results = cv(wineData,[cn2u_learner])
cn2Laplace_results = cv(wineData,[cn2Laplace_learner])



#now we print our metrics, for each result-learner
# As for the required metrics, you can get them using the 'evaluation.scoring' library.
# The 'average' parameter of each metric is used while measuring scores to perform
# a type of averaging on the data. DON'T WORRY MUCH ABOUT THAT JUST YET (AGAIN). USE EITHER
# 'MICRO' OR 'MACRO' (preferably 'macro', at least for final results).
# =============================================================================


# # ADD COMMANDS TO EVALUATE YOUR MODEL HERE (AND PRINT ON CONSOLE)
print("\n ordered learner")
print("Accuracy: " + str(Orange.evaluation.scoring.CA(cn2o_results)))
print("Precision: " + str(Orange.evaluation.scoring.Precision(cn2o_results, average='macro')))
print("F1: " + str(Orange.evaluation.scoring.F1(cn2o_results, average='macro')))
print("Recall: " + str(Orange.evaluation.scoring.Recall(cn2o_results, average='macro')))

print("\n")

print("\n unordered learner")
print("Accuracy: " + str(Orange.evaluation.scoring.CA(cn2u_results)))
print("Precision: " + str(Orange.evaluation.scoring.Precision(cn2u_results, average='macro')))
print("F1: " + str(Orange.evaluation.scoring.F1(cn2u_results, average='macro')))
print("Recall: " + str(Orange.evaluation.scoring.Recall(cn2u_results, average='macro')))

print("\n")

print("\n ordered Laplace learner")
print("Accuracy: " + str(Orange.evaluation.scoring.CA(cn2Laplace_results)))
print("Precision: " + str(Orange.evaluation.scoring.Precision(cn2Laplace_results, average='macro')))
print("F1: " + str(Orange.evaluation.scoring.F1(cn2Laplace_results, average='macro')))
print("Recall: " + str(Orange.evaluation.scoring.Recall(cn2Laplace_results, average='macro')))

print("\n")


# =============================================================================



# Ok, now let's train our learner manually to see how it can classify our data
# using rules.You just want to feed it some data- nothing else.
# =============================================================================


# ADD COMMAND TO TRAIN THE LEARNER HERE
cn2o_classifier = cn2o_learner(wineData)
cn2u_classifier = cn2u_learner(wineData)
cn2Laplace_classifier =  cn2Laplace_learner(wineData)

for rule in cn2o_classifier.rule_list: #orderef list
        print("ordered list")
        print(rule, rule.curr_class_dist.tolist())
print()

for rule in cn2u_classifier.rule_list: #orderef list
        print("unordered list")
        print(rule, rule.curr_class_dist.tolist())
print()

for rule in cn2Laplace_classifier.rule_list: #orderef list
        print("ordered Laplace list")
        print(rule, rule.curr_class_dist.tolist())
print()

