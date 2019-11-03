#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = ki-ocr-spreadsheet
PYTHON_INTERPRETER = python

INCLUDES := -I/usr/local/include -I/usr/lib/libreoffice/sdk/include/ -I/usr/include/opencv4
CPPFLAGS := -std=c++17 -O3 -DCPPU_ENV=gcc3 -DLINUX -DUNX $(INCLUDES)
LDFLAGS :=  -L/usr/local/lib -L/usr/lib/libreoffice/sdk/lib/ -L/usr/lib/libreoffice/program -pthread
GENERATOR_LIBS :=  -luno_cppu -luno_cppuhelpergcc3 -luno_sal -lboost_system -lboost_filesystem
TRIM_LIBS := -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lboost_program_options -lboost_filesystem -lboost_system
SPLIT_LIBS := -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lboost_program_options -lboost_filesystem -lboost_system

GENERATOR = src/data/generate
GENERATOR_SRC = src/data/generate.cpp
GENERATOR_OBJS = $(GENERATOR_SRC:.cpp=.o)

TRIM = src/data/trim
TRIM_SRC = src/data/trim.cpp
TRIM_OBJS = $(TRIM_SRC:.cpp=.o)

SPLIT = src/features/split
SPLIT_SRC = src/features/split.cpp
SPLIT_OBJS = $(SPLIT_SRC:.cpp=.o)


#################################################################################
# COMMANDS                                                                      #
#################################################################################

# make helper 
$(GENERATOR): $(GENERATOR_OBJS)
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(LDFLAGS) $^ $(GENERATOR_LIBS) -o $@

$(TRIM): $(TRIM_OBJS)
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(LDFLAGS) $^ $(TRIM_LIBS) -o $@

$(SPLIT): $(SPLIT_OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ $(SPLIT_LIBS) -o $@

# make dataset
data: $(GENERATOR) $(TRIM) 
	$(PYTHON_INTERPRETER) src/data/generate_data.py

# make features
features: $(SPLIT)
	$(PYTHON_INTERPRETER) src/features/build_features.py

# delete all compiled Python and C++ files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -f $(GENERATOR) $(GENERATOR_OBJS) $(TRIM) $(TRIM_OBJS) $(SPLIT) $(SPLIT_OBJS)

# lint using flake8
lint:
	flake8 src


.PHONY: clean lint data features


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
#
help:
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')

.PHONY: help
