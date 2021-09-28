TESTS = sample_test

CXX = g++
CXXFLAGS = -lgtest -lpthread

$(TESTS):
	$(CXX) -o $@ $@.cpp $(CXXFLAGS)
	./$@

clean:
	rm -rf $(TESTS)
