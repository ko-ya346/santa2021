import itertools
from typing import List


class Solution:
    def __init__(self, split_i1, split_i2, dup_n=3) -> None:
        self.num = "12345671234561723456127345612374561327456137245613742561374526137456213745612347561324756134275613472561347526134756213475612345761234516723451627345162374516234751623457162345176234512673451263745126347512634571263451726345127634512367451236475123645712364517236451273645123764512346751234657123465172346512734651243765124367512436571243651724365127436512473651246375124635712463517246351274635124763512467351426735146273514672351467325146735216473521674352167345216374521634752163457216345271634521764352176453271645327614532764153276451326745132647513264571326451732645137264531726453712645372164537261453726415372645132764531276453217645231764521376452173654217365241736521473652174365217346521736452176345216735421637542163574216354721635427163542176354216735241637524163572416352741635247163524176352416735214673512465371246531724653127465312476531246753142675314627531467253146752316475321647531264753162475316427531647253164752316745321674531267453162745316724531674253167452316754231675243167523416752314675321467531246573124651372465132746513247651324671532467135246713254671235467125346712543671524367154236715432675143267541326754312675432167543261745362174536127453617245361742536174523617453261743526174325617432651742365174263517426531742651374265173426157342617534216753421765342175634217536421753462175342617354261734526173425617342651743261574362157436125743162574312657413265741236574126357412653741265734126574312567413256741235674125367412563741256734125674312576413257614325761342576132457613254761325746132576412357641253761425376124537612543761524376154237615432761543726154376215437612534761253746125376412573641257634125764312574631257436152743615724361574236157432617543621754361275436172543617524361754236175432671543627154367215436712546371254673125476312547361524736154273615472361547326145736214576321475632147653214763521476325147632154763214576231457621345762143576214537621457361245736142573614527361457236145732614753621475361247536142753614725361475236147532614735261473256147326514723651472635147265314726513472651437265147326154736215473612547316254731265471326547123654712635471265347126543716253471625374162537146253716425371624537162543716524371654237165432716543721654371265473125647132564712356471253647125634712564372156437251643275614327564132756431275643217564327156432751643257163425176342516734251637425163472516342751634257163245176324516732451637245163274516324751632457163254716325741632571463275146327154632714563271465327146352714632571643527164357216435712643517264351276435126743512647351264375126435716243517624351672435162743516247351624375162435716423517642351674235164723516427351642375146237514263751423675142376514273651427635142765314276513427651432765142375614235761423567143256714352671435627143567214356712435617243561274356124735612437561243576124356714235617423561472356142735614237516423571643251764325167432516473251643725614372564137256431725643712564731254671324567132465713246751324615732461753246173524617325416723541762354716235476123547621354762315467231546273154623715462317564231576421356742135647213564271356421735624137562413576241356724135627413562471356241735621473562174356217345621735462173564213756421357642153746215374261537421653742156374215367421537642157364215763421576432157642315674231564723156427315642371564231756243157624315672431562743156247315624371562431756234157623415672341562734156237415623471562341756231475623174562317546321745632174653217463521746325174632157463217546312754631725463175246315724631527463152476315246731524637152463175426315742631547263154276315426731542637154263175462315746235174623571462357416235746123574621357462315476235147623541726354172365417235641723546172354167253417625314762531746253176425317624531762543176524317654231765432176543127654317265431762534172653417256341725364172534617253416725431672541367251436725134672153476215347261534721653472156347215364721534672135467213456721346572136457213654721365742136572413657214365721346752136475213674521367542136752413675214376521437562143752614375216437521463725146372154637214563721465372146357214637521436752134672513647251367425136724513672541637254167325417632541736251473625174362517346257136425713624571362547136257413625714362571346275136427513624751362745136275416327541623754126375412367541237654132765413726541376251437625134762513746251376425137624513762541376524137654213765412375641237546132754613725461375246137542613754621375461237541627354126735412763541273654127356412735461273541627534126753412765341275634127536412753461275341627543162754136275143627513462715342671354267134526713425671342657143265714236571426357142653714265731426571342675134267153427615342716534271563427153642715346271354627134562713465271364527136542713652471365274136527143652713462573146257341625734612573462157346251736425173624517362541732654173256417325461732456173246517324615372461532746153247615324167532416573214657321645731264573162457316425731645273165427316524731652743165273416527314652731645723165472316574231657243165723416572314657231645732165473216574321657342165732416537241653274165324716532417653241567321456731245637124563172456312745631247563124576312456731425637142563174256314725631427563142576314256731452637145236714532671453627145367214536712453671425367145237614523716452371465237416523746152347651234765213476523147652341765234716523476152346715234617523461572346152734615237465123746521374652317465237145623714526317452631475263145726314527631452673145627314567231456732154673215647321567432156734215673241563724156327415632471563241756324157632415367241536274153624715362417536241573624153762415326741532647153264175326415732641523764152367415236471523641752364157236415273641526374152634715263417526341572634152763415267341526437152643175264315726431527643152674315264731526413752641357261435726134572613547261357426135724613572641352761435276134527613542761352476135274613527641352674135264713526417352641"

        self.strings = [
            self.num[: split_i1 + dup_n],
            self.num[split_i1 - dup_n : split_i2 + dup_n],
            self.num[split_i2 - dup_n :],
        ]
        self.permutations = [
            "".join(x)
            for x in itertools.permutations(["1", "2", "3", "4", "5", "6", "7"], 7)
        ]

    def get_lens(self) -> List[int]:
        """
        3つの文字列の長さをリストとして返す
        """
        lens = []
        for s in self.strings:
            lens.append(len(s))
        return lens

    def get_ans(self) -> int:
        return max(self.get_lens())

    def check_include(self):
        """
        すべての順列が3つの文字列のいずれかに含まれているか確認
        """
        cnt = 0
        for p in self.permutations:
            tmp = 0
            for string in self.strings:
                # 論理和
                tmp |= p in string
            cnt += tmp ^ 1
        if cnt:
            print("Oh..., Some elements are missing.")
        else:
            print("OK. All elements are inculuded.")