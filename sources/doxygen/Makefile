BEACLS_VERSION = 0.0.1

doc:
	env BEACLS_VERSION=$(BEACLS_VERSION) doxygen Doxyfile.in
	
install:
	cd ../../docs; \
	git add $(BEACLS_VERSION); \
	git commit -m "Updated documents." $(BEACLS_VERSION)

clean:
	
cleanall:
	git rm -r ../../docs/$(BEACLS_VERSION)
	rm -rf -r ../../docs/$(BEACLS_VERSION) 
