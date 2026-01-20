// Lightweight helpers for splitting and counting card vectors
#ifndef CARD_COUNTER_CARD_HELPERS_H
#define CARD_COUNTER_CARD_HELPERS_H

#include <vector>
#include <map>

// Return elements at odd indices (1,3,5...), which this codebase treats as "player" cards
inline std::vector<int> evens(const std::vector<int> &vec){
    std::vector<int> out; 
    for(size_t i = 0; i<vec.size(); ++i){
        if(i%2){
            out.push_back(vec[i]);
        }
    }
    return out; 
}
// inline int EV_stand(double dealer_bust, double player_bust
// }

// Return elements at even indices (0,2,4...), which this codebase treats as "dealer" cards
inline std::vector<int> odds(const std::vector<int> &vec){
    std::vector<int> out; 
    for(size_t i = 0; i<vec.size(); ++i){
        if(i%2==0){
            out.push_back(vec[i]);
        }
    }
    return out; 
}

// Sum the last two elements of a vector. If fewer than two elements exist,
// sum all available elements.
inline int sum_last_two(const std::vector<int> &v){
    int total = 0;
    if(v.size() >= 2){
        total = v[v.size() - 1] + v[v.size() - 2];
    } else {
        for(int x : v) total += x;
    }
    return total;
}

// Count occurrences of card values into a map with keys 1..10 and 11 (aces).
// Any value >10 (except 11) is treated as 10 (face/royal cards).
inline std::map<int,int> count_cards(const std::vector<int> &cards){
    std::map<int,int> counts;
    // initialize keys 1..10 and 11 to 0
    for(int i = 1; i <= 10; ++i) counts[i] = 0;
    counts[11] = 0;

    for(int v : cards){
        int key = 0;
        if(v == 11){
            key = 11; // ace
        } else if(v >= 10){
            key = 10; // tens and royals
        } else if(v >= 1 && v <= 9){
            key = v; // numeric cards 1-9
        } else {
            // ignore out-of-range values
            continue;
        }
        ++counts[key];
    }
    return counts;
}

// Convenience helpers to count only player or dealer cards after splitting
inline std::map<int,int> count_player_cards(const std::vector<int> &played){
    return count_cards(evens(played));
}

inline std::map<int,int> count_dealer_cards(const std::vector<int> &played){
    return count_cards(odds(played));
}

// Return the counts for a full shoe of num_decks decks mapped to our keys:
// keys 2..9 => 4 * num_decks each
// key 10 => 16 * num_decks (10,J,Q,K)
// key 11 => 4 * num_decks (Aces)
// other keys (1) will be 0 to preserve compatibility with existing maps
inline std::map<int,int> full_deck_counts(int num_decks = 1){
    std::map<int,int> full;
    // initialize keys 1..10 and 11 to 0
    for(int i = 1; i <= 10; ++i) full[i] = 0;
    full[11] = 0;

    // standard counts
    for(int v = 2; v <= 9; ++v) full[v] = 4 * num_decks;
    full[10] = 16 * num_decks; // 10,J,Q,K
    full[11] = 4 * num_decks;  // aces
    // key 1 left as 0 (unused)
    return full;
}

// Compute how many of each mapped card value remain in the shoe given seen cards
inline std::map<int,int> remaining_deck_counts(const std::vector<int> &played, int num_decks = 1){
    auto seen = count_cards(played);
    auto full = full_deck_counts(num_decks);
    std::map<int,int> remain;
    for(const auto &p : full){
        int key = p.first;
        int available = p.second - (seen.count(key) ? seen.at(key) : 0);
        if(available < 0) available = 0;
        remain[key] = available;
    }
    return remain;
}


inline std::map<int, double> dealer_probs(int dealer_hand, std::map<int, int> remaining){
    if(dealer_hand > 21){
        return {{22, 1.0}};
    }
    if(dealer_hand >= 17){
        return {{dealer_hand, 1.0}};
    }
    //new sub_map 
    std::map<int,double> sub_map; 
    std::map<int, double> probs; 
    int total_cards_left = 0;
    for (const auto &p : remaining){
        total_cards_left += p.second;
    }
    for(const auto &p : remaining){
        int card = p.first;
        int count = p.second;
        int new_hand = dealer_hand + card; 
        double p_draw = count / total_cards_left;
        remaining[card]--;
        sub_map = dealer_probs(new_hand, remaining);
        remaining[card]++;
        //combine with the chance of new hand 
        for(const auto &p : sub_map){
            probs[p.first] = p.second * p_draw;
        }
    }
   
}



// Small aggregate used for segmented calculations (player/dealer)
struct HandStats{
    int last_two_sum = 0;
    std::map<int,int> counts;

};





// Compute last-two sum and counts for the player (evens) or dealer (odds) portion
inline HandStats compute_hand_stats(const std::vector<int> &played, bool for_player){
    HandStats s;
    if(for_player){
        std::vector<int> hand = evens(played);
        s.last_two_sum = sum_last_two(hand);
        s.counts = count_cards(hand);
    } else {
        // For dealer we only know the face-up card. The user intends this to be the
        // card located between the two last even-indexed cards in the played vector.
        // Find the last two even indices and pick the element between them if present.
        int face_up = 0;
        int last_even_idx = -1;
        int second_last_even_idx = -1;
        for(int i = static_cast<int>(played.size()) - 1; i >= 0; --i){
            if(i % 2 == 0){
                if(last_even_idx == -1) last_even_idx = i;
                else if(second_last_even_idx == -1){
                    second_last_even_idx = i;
                    break;
                }
            }
        }
        if(last_even_idx != -1 && second_last_even_idx != -1){
            int between_idx = second_last_even_idx + 1; // should be odd and between the two evens
            if(between_idx >= 0 && between_idx < static_cast<int>(played.size())){
                face_up = played[between_idx];
            }
        }

        if(face_up != 0){
            std::vector<int> hand = {face_up};
            s.last_two_sum = sum_last_two(hand);
            s.counts = count_cards(hand);
        } else {
            // fallback to full dealer view if we can't determine face-up card
            std::vector<int> hand = odds(played);
            s.last_two_sum = sum_last_two(hand);
            s.counts = count_cards(hand);
        }
    }
    return s;
}

#endif // CARD_COUNTER_CARD_HELPERS_H
