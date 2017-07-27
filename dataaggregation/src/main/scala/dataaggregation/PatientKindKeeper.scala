package dataaggregation

import org.joda.time.DateTime
import sp.EricaEventLogger.StateKeeper
import sp.gPubSub.API_Data.EricaEvent

import scala.collection.mutable

object PatientKindKeeper extends StateKeeper {

  implicit def stringToDateTime(s: String) = DateTime.parse(s)

  val timeOfFirstEvent = mutable.Map[Int, DateTime]() // CareContactId -> DateTime

  val priorities = List("PRIO1", "PRIO2", "PRIO3", "PRIO4", "PRIO5")
  val teams = List("NAKME", "NAKM", "NAKKI", "NAKOR", "NAKBA", "NAKÖN") // "valid clinics" according to ericaBackend code
  val invalidTeams = List("NAKIN", "NAK23T", "NAK29", "NAKKK") // TODO should I include these??
  val patientKinds = List("all", "MEP", "triaged", "metdoctor", "done") ::: priorities ::: teams ::: invalidTeams
  val patientSets = patientKinds.map(str => str -> mutable.Set[Int]()).toMap // sets are of CareContactIds

  val recentlyRemoved = mutable.ListBuffer.fill(500)(-1) // work-around bco events occurring after patient removal, ugh

  override def state(sampleTime: DateTime): List[(String, Int)] = {
    patientKinds.map(kind => kind -> patientSets(kind).size)
  }

  override def handleEvent(ev: EricaEvent): Unit = {
    if(recentlyRemoved.contains(ev.CareContactId)) return () // if event occurs after patient removed, ignore it
    else if(!patientSets("all").contains(ev.CareContactId)) {
      timeOfFirstEvent += ev.CareContactId -> ev.Start
      patientSets("all").add(ev.CareContactId)
    }

    ev.Category match {
      case "RemovedPatient" => {
        timeOfFirstEvent -= ev.CareContactId
        patientSets.foreach(t => t._2.remove(ev.CareContactId))
        recentlyRemoved += ev.CareContactId
        recentlyRemoved.trimStart(1)
      }
      case "Q" => patientSets("all").add(ev.CareContactId)
      //case "T" => patientSets("all").add(ev.CareContactId)
      case "P" => patientSets(ev.Type).add(ev.CareContactId) // Type is here "PRIO4" etc
      case "ReasonForVisitUpdate" if ev.Value == "MEP" => patientSets("MEP").add(ev.CareContactId)
      case "TeamUpdate" => patientSets(ev.Value).add(ev.CareContactId)
      //case s: String if s.contains("removed") => patientsAtNAL.remove(ev.CareContactId)
      case _ => ()
    }
    ev.Type match {
      case "TRIAGE" => patientSets("triaged").add(ev.CareContactId)
      case "LÄKARE" => patientSets("metdoctor").add(ev.CareContactId)
      case "KLAR" => patientSets("done").add(ev.CareContactId)
      case _ => ()
    }
  }

}
