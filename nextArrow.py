def nextpage():
    from pykeyboard import PyKeyboard
    import time
    
    
    k = PyKeyboard()
    time.sleep(0.1)
    # To Create an Alt+Tab combo
    #k.press_key(k.alt_key)
    k.press_key(k.next_key)
    time.sleep(0.05)
    k.release_key(k.next_key)


