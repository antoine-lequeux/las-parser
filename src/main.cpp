#include "cli.hpp"

int main(int argc, const char** argv)
{
    std::locale::global(std::locale("en_US.UTF-8"));
    laspar::launch_cli(argc, argv);

    return 0;
}