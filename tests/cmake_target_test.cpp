#include <tsl/robin_map.h>
#include <tsl/robin_set.h>

int main() {
  tsl::robin_map<int, int> values;
  values[7] = 42;
  tsl::robin_set<int> keys;
  keys.insert(7);

  return values.at(7) == 42 && keys.count(7) == 1 ? 0 : 1;
}
