#!/bin/bash
PROG="../bin/portfolio_simulation"
cd build

cleanup() {
    exit 0
}

trap cleanup SIGINT

while true; do
    echo -e "(1) Build & Compile & Run\n(2) Compile & Run\n(3) Run"
    read -p "Choose an option: " user_input
    if [[ "$user_input" == "1" ]]; then
        if cmake ..; then
            if make; then
                $PROG
            fi
        fi

    elif [[ "$user_input" == "2" ]]; then
        if make; then
            $PROG
        fi
    elif [[ "$user_input" == "3" ]]; then
        $PROG
    else
        echo "Invalid config option"
    fi
done