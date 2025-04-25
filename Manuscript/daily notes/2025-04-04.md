---
tags: DailyNote 
---

# 2025-04-04  10:21

Super frustrating trying to measure spatial frequency.

Max spatial frequency with a 12 pix blur on one channel -- way too jumpy, and the value doesn't mean much.  
![[Pasted image 20250404102258.png]]

Doing things like SF_Var and LSFV look promising, but their range isn't great

LSFV
![[Pasted image 20250404102436.png]]
![[Pasted image 20250404103343.png]]. 

SF_Var
![[Pasted image 20250404102418.png]]

What's also weird is that some blurs give us LSFV differences, but many larger blurs do not -- but responses are definitely affected.





## Questions/tasks 

#todo 

- [ ] look at the RFs for the different blurs
- [ ] confirm the LSFV for a couple RFs
- [ ] does the blur change the mean or norm of the images?  should they be normalized?


