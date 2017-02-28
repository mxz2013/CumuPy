!!! Written by Matteo Guzzo ###
!!! A.D. MMXIV (2014)       ###
      subroutine f2py_calc_spf_mpole(spf,en,nen,prefac,akb,omegapkb,eqp,imeqp,npoles)
      implicit none
      !integer nen,npoles
      !real*8 enexp(0:nen-1),akb(0:npoles-1),omegap(0:npoles-1)
      integer :: nen,npoles
      double precision, dimension (0:nen-1) :: en,spf
      double precision, dimension (0:npoles-1) :: akb,omegapkb
      double precision :: prefac,eqp,imeqp
!Cf2py	intent (in) :: nen
!Cf2py	intent (in) :: en
!Cf2py	intent (in) :: akb
!Cf2py	intent (in) :: omegapkb
!Cf2py	intent(in,out) :: spf
!Cf2py	depend(nen) en
!Cf2py	depend(npoles) en
!Cf2py	depend(npoles) akb
!Cf2py	depend(npoles) omegapkb
      !spf = 0.0d0
      integer :: ien,i,j,k
      double precision :: tmpf1, tmpf2, tmpf3, tmpomp 
      tmpomp = 0.0d0
      spf(:) = 0.0d0
      do ien=0,nen-1
       tmpf1 = 0.0d0
       do i=0,npoles-1
        tmpf2 = 0.0d0
        do j=0,npoles-1
         tmpf3 = 0.0d0
         do k=0,npoles-1
          tmpomp = omegapkb(i)+omegapkb(j)+omegapkb(k)
          tmpf3 = tmpf3 + 1.0d0/3.0d0*akb(k)/((en(ien)-eqp+tmpomp)**2+(imeqp)**2)
         end do
         tmpomp = omegapkb(i)+omegapkb(j)
         tmpf2 = tmpf2 + 1.0d0/2.0d0*akb(j)*(1.0d0/((en(ien)-eqp+tmpomp)**2+(imeqp)**2)+tmpf3)
        end do
        !tmpomp = omegapkb(i)+omegapkb(j)
        tmpf1 = tmpf1 + 1.0d0*akb(i)*(1.0d0/((en(ien)-eqp+omegapkb(i))**2+(imeqp)**2)+tmpf2)
       end do
       !f=prefac*(1./((en(ien)-eqp)**2+(imeqp)**2)+tmpf1)
       spf(ien) = spf(ien) + prefac*(1.0d0/((en(ien)-eqp)**2+(imeqp)**2)+tmpf1)
      end do 
      !return spf
      end subroutine f2py_calc_spf_mpole

      subroutine f2py_calc_crc_mpole(spf,en,nen,bkb,prefac,akb,omegapkb,eqp,imeqp,npoles)
      implicit none
      !integer nen,npoles
      !real*8 enexp(0:nen-1),akb(0:npoles-1),omegap(0:npoles-1)
      integer :: nen,npoles
      double precision, dimension (0:nen-1) :: en,spf
      double precision, dimension (0:npoles-1) :: akb,omegapkb
      double precision :: bkb,prefac,eqp,imeqp
!Cf2py	intent (in) :: nen
!Cf2py	intent (in) :: en
!Cf2py	intent (in) :: akb
!Cf2py	intent (in) :: omegapkb
!Cf2py	intent(in,out) :: spf
!Cf2py	depend(nen) en
!Cf2py	depend(npoles) en
!Cf2py	depend(npoles) akb
!Cf2py	depend(npoles) omegapkb
      !spf = 0.0d0
      integer :: ien,i,j
      double precision :: tmpf1, tmpf2, tmpf3, tmpf4, tmpf5, tmpomp 
      tmpomp = 0.0d0
      spf(:) = 0.0d0
      do ien=0,nen-1
       tmpf1 = 0.0d0
       tmpf3 = 0.0d0
       tmpf4 = 0.0d0
       tmpf5 = 0.0d0
       tmpf2 = 0.0d0
       i = 0
       do j=i+1,npoles-1
!All the comments below are to be removed after testing!!!
        tmpomp = omegapkb(i)+omegapkb(j)
        tmpf5 = bkb/((en(ien)-eqp+tmpomp)**2+(imeqp)**2)
        tmpomp = 2*omegapkb(i)+omegapkb(j)
        tmpf2 = tmpf2 + tmpf5 + 1.0d0*((akb(i)+bkb)**2-akb(i)**2)/((en(ien)-eqp+tmpomp)**2+(imeqp)**2)
       end do
       tmpomp = 3*omegapkb(i)
       tmpf4 = 1.0d0/6.0d0*((akb(i)+bkb)**3-akb(i)**3)/((en(ien)-eqp+tmpomp)**2+(imeqp)**2)
       tmpomp = 2*omegapkb(i)
       tmpf3 = 1.0d0/2.0d0*((akb(i)+bkb)**2-akb(i)**2)/((en(ien)-eqp+tmpomp)**2+(imeqp)**2)
       tmpf1 = tmpf1 + 1.0d0*bkb*(1.0d0/((en(ien)-eqp+omegapkb(i))**2+(imeqp)**2)+tmpf2/2.0d0) + tmpf3 + tmpf4
       do i=1,npoles-1
        tmpf2 = 0.0d0
        do j=0,i-1
         tmpomp = omegapkb(i)+omegapkb(j)
         tmpf5 = bkb/((en(ien)-eqp+tmpomp)**2+(imeqp)**2)
         tmpomp = 2*omegapkb(i)+omegapkb(j)
         tmpf2 = tmpf2 + tmpf5 + 1.0d0*((akb(i)+bkb)**2-akb(i)**2)/((en(ien)-eqp+tmpomp)**2+(imeqp)**2)
        end do
        do j=i+1,npoles-1
         tmpomp = omegapkb(i)+omegapkb(j)
         tmpf5 = bkb/((en(ien)-eqp+tmpomp)**2+(imeqp)**2)
         tmpomp = 2*omegapkb(i)+omegapkb(j)
         tmpf2 = tmpf2 + tmpf5 + 1.0d0*((akb(i)+bkb)**2-akb(i)**2)/((en(ien)-eqp+tmpomp)**2+(imeqp)**2)
        end do
        tmpomp = 3*omegapkb(i)
        tmpf4 = 1.0d0/6.0d0*((akb(i)+bkb)**3-akb(i)**3)/((en(ien)-eqp+tmpomp)**2+(imeqp)**2)
        tmpomp = 2*omegapkb(i)
        tmpf3 = 1.0d0/2.0d0*((akb(i)+bkb)**2-akb(i)**2)/((en(ien)-eqp+tmpomp)**2+(imeqp)**2)
        tmpf1 = tmpf1 + 1.0d0*bkb*(1.0d0/((en(ien)-eqp+omegapkb(i))**2+(imeqp)**2)+tmpf2/2.0d0) + tmpf3 + tmpf4
       end do
       !f=prefac*(1./((en(ien)-eqp)**2+(imeqp)**2)+tmpf1)
       !spf(ien) = spf(ien) + exp(-bkb)*prefac*tmpf1
       spf(ien) = spf(ien) + prefac*tmpf1
      end do 
      !return spf
      end subroutine f2py_calc_crc_mpole

      subroutine f2py_calc_spf_mpole_extinf(spf,en,nen,prefac,akb,omegapkb,wkb,eqp,imeqp,npoles)
      implicit none
      !integer nen,npoles
      !real*8 enexp(0:nen-1),akb(0:npoles-1),omegap(0:npoles-1)
      integer :: nen,npoles
      double precision, dimension (0:nen-1) :: en,spf
      double precision, dimension (0:npoles-1) :: akb,omegapkb,wkb
      double precision :: prefac,eqp,imeqp
!Cf2py	intent (in) :: nen
!Cf2py	intent (in) :: en
!Cf2py	intent (in) :: akb
!Cf2py	intent (in) :: omegapkb
!Cf2py	intent(in,out) :: spf
!Cf2py	depend(nen) en
!Cf2py	depend(npoles) en
!Cf2py	depend(npoles) akb
!Cf2py	depend(npoles) omegapkb
!Cf2py	depend(npoles) wkb
      !spf = 0.0d0
      integer :: ien,i,j,k
      double precision :: tmpf1, tmpf2, tmpf3, tmpomp, tmpgamma 
      tmpomp = 0.0d0
      tmpgamma = 0.0d0 ! This is a double width
      spf(:) = 0.0d0
      do ien=0,nen-1
       tmpf1 = 0.0d0
       do i=0,npoles-1
        tmpf2 = 0.0d0
        do j=0,npoles-1
         tmpf3 = 0.0d0
         do k=0,npoles-1
          tmpomp = omegapkb(i)+omegapkb(j)+omegapkb(k)
          tmpgamma = (wkb(i)+wkb(j)+wkb(k))/2
          tmpf3 = tmpf3 + 1.0d0/3.0d0*akb(k)/((en(ien)-eqp+tmpomp)**2+(tmpgamma+imeqp)**2)
         end do
         tmpomp = omegapkb(i)+omegapkb(j)
         tmpgamma = (wkb(i)+wkb(j))/2
         tmpf2 = tmpf2 + 1.0d0/2.0d0*akb(j)*(1.0d0/((en(ien)-eqp+tmpomp)**2+(tmpgamma+imeqp)**2)+tmpf3)
        end do
        tmpomp = omegapkb(i)
        tmpgamma = (wkb(i))/2
        tmpf1 = tmpf1 + 1.0d0*akb(i)*(1.0d0/((en(ien)-eqp+tmpomp)**2+(tmpgamma+imeqp)**2)+tmpf2)
       end do
       !f=prefac*(1./((en(ien)-eqp)**2+(imeqp)**2)+tmpf1)
       spf(ien) = spf(ien) + prefac*(1.0d0/((en(ien)-eqp)**2+(imeqp)**2)+tmpf1)
       !spf(ien) = 1.0d0
      end do 
      !spf(:) = 0.0d0
      !return spf
      end subroutine f2py_calc_spf_mpole_extinf
