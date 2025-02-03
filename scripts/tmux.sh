tmux new-session -s my_session \; \
    new-window \; \
    new-window \; \
    new-window \; \
    new-window \; \
    new-window \; \
    split-window -v -t 3 \; \
    send-keys -t 3.1 'htop' C-m \; \
    send-keys -t 0.0 'vim -p *.py' C-m \; \
    send-keys -t 3.0 'nvidia-smi -l 4' C-m \; \
    select-window -t 0 \; \
    attach-session -t my_session
