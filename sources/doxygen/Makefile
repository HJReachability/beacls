BEACLS_VERSION = 0.0.1

doc:
	env BEACLS_VERSION=$(BEACLS_VERSION) doxygen Doxyfile.in
	
install:
	git add ../../doc/$(BEACLS_VERSION)
	git commit -m "Updated documents." ../../doc/$(BEACLS_VERSION)

clean:
	
cleanall:
	git rm -r ../../doc/$(BEACLS_VERSION)
	rm -rf -r ../../doc/$(BEACLS_VERSION) 
