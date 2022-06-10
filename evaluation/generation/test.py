text = ''
while True:
        try:
            dummy = raw_input("""Enter the paragraph :""")
            text += dummy
        except KeyboardInterrupt:
            print(text)
            text = ''
