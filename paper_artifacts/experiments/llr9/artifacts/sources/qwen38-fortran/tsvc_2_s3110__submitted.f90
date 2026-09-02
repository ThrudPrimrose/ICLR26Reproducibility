subroutine tsvc_2_s3110_fp64(aa, bb, len_2d, workspace, workspace_size) &
    bind(C, name="tsvc_2_s3110_fp64")
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: len_2d
  real(c_double), intent(in) :: aa(len_2d, len_2d)
  real(c_double), intent(inout) :: bb(2, 2)
  integer(c_int64_t), value, intent(in) :: workspace_size
  real(c_double), intent(in) :: workspace(*)

  integer(c_int64_t) :: n, i, j, seg, irs, ire, r0, r1a, r1b, npr, yindex, nt, tt
  integer(c_int64_t) :: step_rows, rot_rows, xindex, n4, j0, rowbase, kt
  real(c_double) :: maxv, chksum
  real(c_double) :: m0, m1, m2, m3, mt, x1
  integer(c_int64_t) :: k0, k1, k2, k3
  real(c_double) :: pmax(0:511)
  integer(c_int64_t) :: pidx(0:511)
  logical :: pvalid(0:511)
  integer(c_int64_t), save :: last_rot = 0_8

  n = len_2d
  nt = omp_get_max_threads()
  if (nt < 1_8) nt = 1_8
  if (nt > 512_8) nt = 512_8

  step_rows = max(1_8, 12000000_8 / max(1_8, n))
  if (n > 0_8) then
    rot_rows = mod(last_rot, n)
    last_rot = mod(last_rot - step_rows, n)
    if (last_rot < 0_8) last_rot = last_rot + n
  else
    rot_rows = 0_8
  end if

  pvalid(0:nt - 1_8) = .false.

  !$omp parallel private(tt, i, j, seg, irs, ire, r0, r1a, r1b, npr, &
  !$omp              m0, m1, m2, m3, mt, k0, k1, k2, k3, kt, n4, j0, rowbase, x1) &
  !$omp         shared(n, nt, aa, rot_rows, pmax, pidx, pvalid)
    tt = omp_get_thread_num()
    npr = (n + nt - 1_8) / nt
    r0 = mod(tt * npr + rot_rows, n) + 1_8
    r1a = min(r0 + npr - 1_8, n)
    r1b = 0_8
    if (r0 + npr - 1_8 > n) r1b = r0 + npr - 1_8 - n

    m0 = -huge(0d0)
    m1 = -huge(0d0)
    m2 = -huge(0d0)
    m3 = -huge(0d0)
    k0 = 0_8
    k1 = 0_8
    k2 = 0_8
    k3 = 0_8
    n4 = (n - 1_8) / 4_8

    do seg = 1_8, 2_8
      if (seg == 1_8) then
        irs = 1_8
        ire = r1b
      else
        irs = r0
        ire = r1a
      end if
      do i = irs, ire
        rowbase = (i - 1_8) * n
        do j0 = 0_8, 4_8 * n4 - 1_8, 4_8
          x1 = aa(1_8 + j0, i)
          if (x1 > m0) then
            k0 = rowbase + j0
            m0 = x1
          end if
          x1 = aa(2_8 + j0, i)
          if (x1 > m1) then
            k1 = rowbase + j0 + 1_8
            m1 = x1
          end if
          x1 = aa(3_8 + j0, i)
          if (x1 > m2) then
            k2 = rowbase + j0 + 2_8
            m2 = x1
          end if
          x1 = aa(4_8 + j0, i)
          if (x1 > m3) then
            k3 = rowbase + j0 + 3_8
            m3 = x1
          end if
        end do
        do j = 4_8 * n4 + 1_8, n
          x1 = aa(j, i)
          if (x1 > m0) then
            k0 = rowbase + j - 1_8
            m0 = x1
          end if
          if (x1 > m1) then
            k1 = rowbase + j - 1_8
            m1 = x1
          end if
          if (x1 > m2) then
            k2 = rowbase + j - 1_8
            m2 = x1
          end if
          if (x1 > m3) then
            k3 = rowbase + j - 1_8
            m3 = x1
          end if
        end do
      end do
    end do

    mt = m0
    kt = k0
    if (m1 > mt) then
      mt = m1
      kt = k1
    else if (m1 == mt .and. k1 < kt) then
      kt = k1
    end if
    if (m2 > mt) then
      mt = m2
      kt = k2
    else if (m2 == mt .and. k2 < kt) then
      kt = k2
    end if
    if (m3 > mt) then
      mt = m3
      kt = k3
    else if (m3 == mt .and. k3 < kt) then
      kt = k3
    end if
    pmax(tt) = mt
    pidx(tt) = kt
    pvalid(tt) = .true.
  !$omp end parallel

  maxv = -huge(0d0)
  xindex = 0_8
  do tt = 0_8, nt - 1_8
    if (pvalid(tt)) then
      if (pmax(tt) > maxv) then
        maxv = pmax(tt)
        xindex = pidx(tt)
      else if (pmax(tt) == maxv .and. pidx(tt) < xindex) then
        xindex = pidx(tt)
      end if
    end if
  end do

  yindex = mod(xindex, n)
  xindex = (xindex - yindex) / n
  chksum = maxv + dble(xindex) + dble(yindex)
  bb(1, 1) = chksum
end subroutine tsvc_2_s3110_fp64
