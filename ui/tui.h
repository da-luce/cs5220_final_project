#pragma once
#include <string>
#include <vector>
#include <utility>

namespace tui {

// Initializes ncurses, sets up color pairs and standard terminal config
void init_ncurses();

// Prompts the user with a list of choices.
// Returns the index of the selected choice, -1 if the user chose to go back, or -2 to quit.
int prompt_list(const std::string& title, 
                const std::vector<std::pair<std::string, std::string>>& history, 
                const std::string& question, 
                const std::vector<std::string>& choices);

// Prompts the user to enter a text string.
// Returns 1 for success, -1 if the user cancelled/went back, or -2 to quit.
int prompt_input(const std::string& title,
                 const std::vector<std::pair<std::string, std::string>>& history,
                 const std::string& question,
                 std::string& out_str);

// Displays an error message and waits for user input
void show_error(const std::string& msg);

// A declarative Immediate Mode GUI wrapper for wizard prompts
class Form {
private:
    std::string title;
    int current_step = 0;
    int render_step = 0;
    bool _running = true;
    std::vector<std::pair<std::string, std::string>> history;
    std::vector<std::string> answers;

    void reset_render() {
        render_step = 0;
        history.clear();
    }

public:
    Form(const std::string& t) : title(t) {}
    
    bool running() const { return _running; }
    
    std::string select(const std::string& question, const std::vector<std::string>& choices) {
        if (render_step < current_step) {
            history.push_back({question, answers[render_step]});
            return answers[render_step++];
        }
        if (render_step == current_step) {
            int res = prompt_list(title, history, question, choices);
            if (res >= 0) {
                if (current_step < (int)answers.size()) answers[current_step] = choices[res];
                else answers.push_back(choices[res]);
                current_step++;
                reset_render();
            } else if (res == -1) {
                if (current_step > 0) current_step--;
                reset_render();
            } else {
                _running = false;
            }
        }
        return "";
    }

    std::string text_input(const std::string& question) {
        if (render_step < current_step) {
            history.push_back({question, answers[render_step]});
            return answers[render_step++];
        }
        if (render_step == current_step) {
            std::string out;
            int res = prompt_input(title, history, question, out);
            if (res == 1) {
                if (current_step < (int)answers.size()) answers[current_step] = out;
                else answers.push_back(out);
                current_step++;
                reset_render();
            } else if (res == -1) {
                if (current_step > 0) current_step--;
                reset_render();
            } else {
                _running = false;
            }
        }
        return "";
    }

    void set_error(const std::string& msg) {
        show_error(msg);
        if (current_step > 0) current_step--;
        reset_render();
    }
};

} // namespace tui