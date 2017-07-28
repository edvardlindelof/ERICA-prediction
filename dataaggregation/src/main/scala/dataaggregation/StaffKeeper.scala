package dataaggregation

import org.joda.time.DateTime
import sp.EricaEventLogger.StateKeeper
import sp.gPubSub.API_Data.EricaEvent

import scala.collection.mutable

object StaffKeeper extends StateKeeper {

  implicit def stringToDateTime(s: String) = DateTime.parse(s)

  val minsToLookForIds = 60 // i.e. number of unique doctor ids seen in past 60 mins is used as number of doctors

  var recentDoctorEvents = mutable.ListBuffer[(String, DateTime)]() // String is personal doctor id
  var recentTeamEvents = mutable.ListBuffer[(String, DateTime)]() // String is team name e.g. NAKM

  override def handleEvent(ev: EricaEvent): Unit = {
    if(ev.Type == "LÄKARE") recentDoctorEvents += ((ev.Value, ev.Start))
    if(ev.Category == "TeamUpdate") recentTeamEvents += ((ev.Value, ev.Start))

    val tooOld = (dt: DateTime) => ev.Start.minusMinutes(minsToLookForIds).isAfter(dt)
    recentDoctorEvents = recentDoctorEvents.dropWhile(t => tooOld(t._2))
    recentTeamEvents = recentTeamEvents.dropWhile(t => tooOld(t._2))
  }

  override def state(sampleTime: DateTime): List[(String, Int)] = {
    val nDoctors = recentDoctorEvents.map(_._1).toSet.size
    val nTeams = recentTeamEvents.map(_._1).toSet.size
    //println(recentDoctorEvents.map(_._1))
    //println(recentTeamEvents.map(_._1))
    ("doctors" + minsToLookForIds, nDoctors) :: ("teams" + minsToLookForIds, nTeams) :: Nil
  }

}
