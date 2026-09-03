# Reviewer comments from A D

Source: `Master Thesis - Pei Ling Song_AD.pdf`

Notes:

- All 49 comments are listed in PDF page order.
- Page numbers below are PDF page numbers, not necessarily the printed thesis page numbers.
- Reviewer wording is preserved verbatim, including spelling and grammar.
- "Highlighted text" is omitted for free-standing text-note annotations.

## PDF page 9

- [ ] **01 - Background placement**
  - Highlighted text: "In the TABULA-NL configuration adopted in this study, building type and the construction period derived from construction year jointly determine the archetype cell, while floor count serves as an additional feature in the subsequent energy classification."
  - Comment: "this doesn't belong to background"

## PDF page 11

- [ ] **02 - Clarify assessment scope**
  - Highlighted text: "The assessment is limited to the evaluated population, tasks, and geographic setting."
  - Comment: "what do you mean in this sentence?"

- [ ] **03 - Extraction versus prediction**
  - Highlighted text: "extraction"
  - Comment: "prediction?"

- [ ] **04 - Objectives read as steps**
  - Highlighted text: "objectives:"
  - Comment: "as described below it sounds more like steps, not objectives"

- [ ] **05 - Explain the Amsterdam exclusion**
  - Highlighted text: "The analysis also estimates the marginal contribution of individual attributes and evaluates transfer to Amsterdam after excluding the city from model development."
  - Comment: "You speak about removing Amsterdam for the second time in this thesis without any explanation, why and what for."

## PDF page 13

- [ ] **06 - Support required attributes with more literature**
  - Highlighted text: "If building type, construction year, or floor count is unavailable, this route does not show how those attributes should be recovered or how replacing them with inferred values affects downstream performance."
  - Comment: "You haven't defined that these exact attributes are required in these methods. And you define this category based on only one paper. Can you find more papers using similar approach?"

- [ ] **07 - Add a transition between sentences**
  - Highlighted text: "of the individual building itself. German census data"
  - Comment: "missing link between the two sentences"

- [ ] **08 - Soften the transferability claim**
  - Highlighted text: "Proxy-based methods therefore offer a feasible route for large-scale data completion, but their transferability remains dependent on proxy resolution, geographic context, and target taxonomy."
  - Comment: "It's a bit unfair to create this kind of statements based only on two papers. Maybe you could soften it a bit."

## PDF page 14

- [ ] **09 - Reconsider the UK-led sentence opening**
  - Highlighted text: ". In the United Kingdom,"
  - Comment: "is it important that it was in the UK? Starting a sentence from the country reads like a newspaper article."

- [ ] **10 - Verify whether sources mention TABULA**
  - Highlighted text: "Moreover, direct models bypass explicit TABULA parameters."
  - Comment: "do any of these sources mention TABULA?"

## PDF page 15

- [ ] **11 - Cite OpenFACADES by author**
  - Highlighted text: "OpenFACADES"
  - Comment: "this is a name of the tool, better to put the authors here"

- [ ] **12 - Define frozen backbone**
  - Highlighted text: "frozen-backbone"
  - Comment: "you should explain what does it mean that a model is frozen"

- [ ] **13 - Replace vague cross-reference**
  - Highlighted text: "above"
  - Comment: "better use the reference, like reviewed in Section X.X."

- [ ] **14 - Continue the zero-shot explanation**
  - Highlighted text: "Furthermore, the term 'zero-shot' does not necessarily mean that development examples played no role in prompt selection."
  - Comment: "continue the thought"

## PDF page 17

- [ ] **15 - Define TABULA-NL**
  - Highlighted text: "TABULA-NL"
  - Comment: "You should explain what is \"TABULA-NL\". Is this a Dutch subset of the TABULA dataset or is this the name of your dataset with all the information that you collected at the begining of the work?"

- [ ] **16 - Remove unnecessary retained-information wording**
  - Highlighted text: "and retains information on image source, view quality, and pairing status."
  - Comment: "You don't have to mention that information was retained during a workflow step. It should be retained by default, unless it's explicitly redundant, right?"

- [ ] **17 - Increase Figure font size or change orientation**
  - Comment: "This is a really nice figure, but the fonts are much too small. Generally text in figures should not be much smaller than the text in the manuscript. Maybe it would make sense to make this pipeline vertical?"

## PDF page 18

- [ ] **18 - Define semi-synthetic dataset**
  - Highlighted text: "semi-synthetic dataset"
  - Comment: "you should also explain what semi-synthetic means"

- [ ] **19 - Move archetype-grounded material to the literature review**
  - Comment: "you could also do this in the literature review section \"archetype-grounded...\""

- [ ] **20 - Increase font size**
  - Comment: "font size"

- [ ] **21 - Define synthetic data earlier**
  - Comment: "and even earlier, what does it mean that data is synthetic?"

## PDF page 19

- [ ] **22 - Add a source for every dataset**
  - Highlighted text: "BAG, 3DBAG, EP-Online, Mapillary, and TABULA-NL."
  - Comment: "add sources to each"

- [ ] **23 - Introduce EP-Online energy classes before merging labels**
  - Highlighted text: "Ratings from A+ through A++++ are merged into class A."
  - Comment: "It reads like there was a sentence missing before this. So you say that you use the energy classes from the EP-Online. What are these classes, what values they have? You go directly to merging A+ through A++++ without any explanation why. There are many similar places in the manuscript, where I feel like a sentence was missing. Here I imagine something like \"These classes evaluate building energy performance using letters from A to G, however the super performing buildings can get classes from A+ to A+++ if they perform super good\". Ofc don't copy this - that's just an example what I feel is missing in this specific part."

## PDF page 21

- [ ] **24 - Remove wordy or unnecessary clarification**
  - Highlighted text: "This centroid containment operation determined which footprints proceeded to image processing. It did not establish the final link between the SVI and reference data."
  - Comment: "And then we have sentences like these that are super wordy and unnecessary. Of course it did not establish this link - why should it?"

- [ ] **25 - Clarify whether the identifier was later removed**
  - Highlighted text: "Each crop initially retained the OpenFACADES building identifier associated with its source footprint."
  - Comment: "Initially retained and then removed?"

## PDF page 23

- [ ] **26 - Clarify image counts and dataset comparability**
  - Highlighted text: "number of images per building,"
  - Comment: "Do they? I thought you compare the same dataset by all the models."

## PDF page 24

- [ ] **27 - Introduce the model configurations earlier**
  - Highlighted text: "The two configurations differ in the parameters updated during training."
  - Comment: "This is a very good description, anjd I have a feeling that it was a bit missing before. You could write a short version of this additionally by the moment you introduce the models."

## PDF page 25

- [ ] **28 - Check tense consistency throughout the thesis**
  - Highlighted text: "this study evaluates"
  - Comment: "please check for consistency where you use a present and past tense in the entire script."

## PDF page 26

- [ ] **29 - Clarify whether the figure was AI-generated**
  - Comment: "was this figure generated by an AI?"

## PDF page 28

- [ ] **30 - Add original Dutch category terms or explain translations**
  - Highlighted text: "Table 3.4: Operational mapping from retained EP-Online residential building categories to"
  - Comment: "this is actually a place where you could include the original Dutch terms in this table (as a third column). Or are these official translations from the database?"

## PDF page 29

- [ ] **31 - Rewrite wording that sounds AI-generated**
  - Highlighted text: "rather than an observation of the building's current envelope condition."
  - Comment: "this sentence sounds like GPT"

## PDF page 30

- [ ] **32 - Add a small diagram of the fitted-classifier workflow**
  - Highlighted text: "For each task, the final downstream classifier is fitted once on the complete development set using reference features. The same fitted classifier is then applied separately to the reference-derived and vision-derived holdout features."
  - Comment: "can you make a tiny diagram for this?"

## PDF page 32

- [ ] **33 - Increase figure font size**
  - Comment: "great figure, but increase the font size"

- [ ] **34 - Unify 'routes' and 'route families' terminology**
  - Highlighted text: "four evaluation routes"
  - Comment: "Please unify the vocabulary used across the entire paper. The reader now must figure out what do you mean by \"Routes\" and \"Route Families\"."

- [ ] **35 - Unify 'routes' and 'configurations' terminology**
  - Comment: "previously you called this \"configurations\" right?"

## PDF page 33

- [ ] **36 - Explain the shared fitted-model protocol more clearly**
  - Highlighted text: "The M1 and M3 attribution comparison uses a shared fitted-model protocol."
  - Comment: "It's really difficult to understand what you mean by this sentence. Try to be more descriptive, imagine being a teacher trying to explain the results to the students."

- [ ] **37 - Reduce unexplained technical vocabulary in Results**
  - Highlighted text: "The five development folds support model selection"
  - Comment: "you are using a lot of scientific vocabulary, probably partly unnecessarily. You spoke about the stratified split in the method, but people won't remember it, and seeing \"fold\" here will most probably ask themselves \"what fold means here\"? Also, you are now introducing the OOF, while you shouldn't introduce new terms in the results."

## PDF page 36

- [ ] **38 - Clarify the whole-building transmission coefficient and cite the equation**
  - Highlighted text: "For building i, the transmission heat transfer coefficient is defined as"
  - Comment: "typically we mean just u-value by this term. Here I see you unify it to represent the entire building as one value. Can you give a source of this equation?"

- [ ] **39 - Define signed bias and explain why it is used**
  - Highlighted text: "signed bias"
  - Comment: "you have to explain the term and why not absolute bias, as you present the formula. You wrote \"retains the direction\", but this sounds enigmatic. A direction of what?"

## PDF page 45

- [ ] **40 - Add facade examples and model predictions to the Results**
  - Comment: "What I am missing in this section is visuals. You are speaking about classification of facade properties. Can you shoq a few examples of a facade and show how the models performed? The charts are really good, but the reader can totally forget that you are speaking about building facades. Also, this would also present the core challenge - the facades may be obstructed, ht perspective may be different to work with. A few examples along with the corresponding predictions would definitely benefit the paper."

## PDF page 48

- [ ] **41 - Add a Discussion section and move relevant material into it**
  - Highlighted text: "The analysis characterises a sample composition effect rather than a formal upper bound on seven-class performance."
  - Comment: "The Discussion section is missing, and the section above can be a part of it."

## PDF page 49

- [ ] **42 - Broaden the framing of the main goal**
  - Highlighted text: "This thesis assessed whether building attributes inferred from SVI can substitute for reference attributes in a TABULA-based building stock enrichment workflow."
  - Comment: "Is the main overarching goal to use the TABULA? I would put a broader context here, like generally how SVI models can extract attributes that would be useful for energy class prediction. And then that this was answqered using the case of tabula dataset and enrichment workflow"

## PDF page 50

- [ ] **43 - Positive feedback on Contributions section**
  - Highlighted text: "5.2 Contributions"
  - Comment: "well made"

## PDF page 51

- [ ] **44 - Discuss buildings linked to multiple certificates**
  - Highlighted text: "linked to multiple certificates that differ across units or registration dates."
  - Comment: "This is a big problem and it would be good if you could explain, how this influences your project. Do you have the same building twice, with two different labels? I mean, explain this in the discussion, where you mention this problem for the first time. Here a short reference is enough, as you have now."

- [ ] **45 - Move Limitations into the Discussion**
  - Highlighted text: "The"
  - Comment: "actually the Limitations would match the Discussion section better."

## PDF page 52

- [ ] **46 - Positive feedback on construction-year outcome**
  - Highlighted text: "Further development of visual attribute extraction should prioritise construction year because it made the largest marginal contribution to downstream energy class classification among the evaluated attributes."
  - Comment: "very good outcome thought"

- [ ] **47 - Move Future Work into the Discussion**
  - Highlighted text: "5.4 Future Work"
  - Comment: "and the future work also to the discussion section"

## PDF page 57

- [ ] **48 - Correct wording around Table B.1**
  - Highlighted text: "training procedure as table B.1,"
  - Comment: "wording. Table B.1. doesn't follow the training procedure."

## PDF page 61

- [ ] **49 - Add a generative-AI declaration if applicable**
  - Comment: "If you used generative AI anywhere in the work, you should declare it. See the TUM citation guide."
