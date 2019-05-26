def previousPage():    
    from pykeyboard import PyKeyboard
    import time
    
    
    k = PyKeyboard()
    time.sleep(0.1)
    
    
    
    k.press_key(k.left_key)
    time.sleep(0.05)
    k.release_key(k.left_key)
