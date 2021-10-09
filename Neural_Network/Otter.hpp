//
//  Otter.hpp
//  Neural_Network
//
//  Created by 陳均豪 on 2021/10/9.
//

#ifndef Otter_hpp
#define Otter_hpp

#include <stdio.h>
#include <string>
#include <vector>
#include <fstream>

using namespace std;

#define EARSE_CHARACTER(str, c) str.erase(std::remove_if(str.begin(), str.end(), [](unsigned char x) { return x == c; }), str.end());

#define EARSE_SPACE(str) EARSE_CHARACTER(str, ' ')

#define WRITE_SPACE(file, n) for (int i = n; i--; ) fprintf(file, " ");

struct Stick {
    Stick(string type_, string info_) : type(type_), info(info_) {}
    string type;
    string info;
};

typedef Stick Option;

class Otter {
public:
    Otter(string name_) : name(name_) {}
    bool parse_blueprint(fstream &blueprint);
    void save_blueprint(FILE *project, int format = 0);
    void addPartner(Otter partner_) {partner.push_back(partner_);}
    void addMaterial(Stick element) {material.push_back(element);}
    string getName() {return name;}
    vector<Stick> getMaterial();
    bool idle() {return material.empty();}
private:
    string name;
    vector<Otter> partner;
    vector<Stick> material;
};

class Otter_Leader {
public:
    Otter_Leader(string name = "") : project_name(name) {}
    bool read_project(const char *project_name);
    bool save_project(const char *project_name);
    bool find_team(fstream &project, string &team_name);
    void addTeam(Otter team) {teams.push_back(team);}
    void addOption(Option opt) {option.push_back(opt);}
    string getName() {return project_name;}
    string getTeamName(int index) {return teams[index].getName();}
    int members() {return (int)teams.size();}
    vector<Option> getOption() {return option;}
    vector<Stick> getMaterial(int index);
private:
    string project_name;
    vector<Option> option;
    vector<Otter> teams;
};

Option parse_option(fstream &f);
string parse_arg(fstream &f);

#endif /* Otter_hpp */
